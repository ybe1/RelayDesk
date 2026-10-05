"""Order facts precede policy retrieval; selection pauses without side effects."""
import asyncio
import json
from uuid import uuid4
from langchain_core.messages import AIMessage,HumanMessage
from langgraph.graph import StateGraph,START,END
from langgraph.types import interrupt
from langgraph.runtime import Runtime
from langgraph.config import get_stream_writer
from .contracts import WorkflowState,TurnContext
from .budget import BudgetExceeded
from .bare_loop import tool_error
from .agent import persist
from .expansion import expand_queries
from .order_context import demo_order_options
from .retrieval import CitationRegistry,gate_result,calibrated_gate,retrieve_policy_queries
from app.services.refunds import validate_refund_defaults


def refund_offer_defaults(state):
    if state.get('intent')!='退款退货' or not state.get('gate_allowed'): return None
    defaults={key:state.get(key) for key in ('order_id','order_snapshot','refund_assessment','evidence')}
    try: return validate_refund_defaults(defaults)
    except ValueError: return None


def build_after_sales_graph(gate_threshold=None):
    from app.config import settings
    threshold=gate_threshold if gate_threshold is not None else settings.workflow_gate_threshold
    def trace(state,node):
        get_stream_writer()(('workflow_status',{'node':node}))
        return {'node_trace':state.get('node_trace',[])+[node]}
    def slot(state): return trace(state,'slot')
    def prepare_selection(state,runtime:Runtime[TurnContext]):
        selection={'selection_id':uuid4().hex,'turn_id':state['turn_id'],'options':demo_order_options()}
        return {**trace(state,'prepare_selection'),'selection':selection,'budget_usage':runtime.context.budget.snapshot()}
    def select_order(state):
        selected=interrupt(state['selection'])
        if not isinstance(selected,dict) or selected.get('selection_id')!=state['selection']['selection_id']:
            raise ValueError('stale selection')
        order_id=selected.get('order_id')
        if order_id not in {item['order_id'] for item in state['selection']['options']}: raise ValueError('invalid selection')
        return {**trace(state,'select_order'),'order_id':order_id,'selection':{},'messages':[
            HumanMessage(content='选择订单'+order_id,id=f"{state['turn_id']}:selection:{state['selection']['selection_id']}")]}
    async def fetch_order(state,runtime:Runtime[TurnContext]):
        context=runtime.context
        update=trace(state,'fetch_order')
        call={'name':'query_order','args':{'order_id':state['order_id']},'id':state['turn_id'][:24]+'-order'}
        request=AIMessage(content='',tool_calls=[call],id=state['turn_id']+':pre-order')
        await persist(context,request)
        get_stream_writer()(('tool_status',{'tool_name':'query_order','phase':'running','tool_call_id':call['id']}))
        try:
            async with asyncio.timeout(context.budget.remaining_seconds()):
                observation=await context.agent.executor.execute(call,before_attempt=context.budget.reserve_tool)
        except (BudgetExceeded,TimeoutError) as exc: observation=tool_error(call,str(exc))
        observation.id=state['turn_id']+':tool:'+call['id']
        await persist(context,observation)
        update['messages']=[request,observation]
        try:
            payload=json.loads(observation.content)
            data=payload['result']
            data=json.loads(data) if isinstance(data,str) else data
            if not isinstance(data,dict) or data.get('order_id')!=state['order_id']: raise ValueError('wrong order result')
            update['order_snapshot']=data
        except (KeyError,TypeError,ValueError):
            update.update(order_snapshot={},gate_allowed=False,gate_reason='order_unavailable')
        get_stream_writer()(('tool_status',{'tool_name':'query_order','phase':'success' if update.get('order_snapshot') else 'failed','tool_call_id':call['id']}))
        update['budget_usage']=context.budget.snapshot()
        return update
    async def expand(state,runtime:Runtime[TurnContext]):
        context=runtime.context
        update=trace(state,'expand')
        try:
            update['expanded_queries']=await expand_queries(context.agent.model,state['question'],state['order_snapshot'],state['intent'],context.budget)
        except Exception as exc: update.update(expanded_queries=[],gate_allowed=False,gate_reason='expansion_unavailable:'+type(exc).__name__)
        update['budget_usage']=context.budget.snapshot()
        return update
    async def policy_retrieval(state,runtime:Runtime[TurnContext]):
        context=runtime.context
        update=trace(state,'policy_retrieval')
        try:
            result=await (context.retrieve_queries or retrieve_policy_queries)(state['question'],state['expanded_queries'],context.budget)
            if context.confidence_config is not None:
                snapshot,gate=calibrated_gate(state['question'],result,context.confidence_config,
                    query_sources=state.get('expanded_queries',[]))
                update.update(evidence_snapshot=snapshot.model_dump(mode='json'),confidence=gate.model_dump())
            else:
                gate=gate_result(result,threshold)
            registry=CitationRegistry()
            update.update(evidence=registry.merge(result.hits) if gate.allowed else [],citations=registry.citations,
                          gate_allowed=gate.allowed,gate_reason=gate.reason)
        except Exception as exc: update.update(evidence=[],citations=[],gate_allowed=False,gate_reason='retrieval_unavailable:'+type(exc).__name__)
        update['budget_usage']=context.budget.snapshot()
        return update
    builder=StateGraph(WorkflowState,context_schema=TurnContext)
    for name,node in [('slot',slot),('prepare_selection',prepare_selection),('select_order',select_order),
                      ('fetch_order',fetch_order),('expand',expand),('policy_retrieval',policy_retrieval)]: builder.add_node(name,node)
    builder.add_edge(START,'slot')
    builder.add_conditional_edges('slot',lambda s:'fetch_order' if s.get('order_id') else 'prepare_selection')
    builder.add_edge('prepare_selection','select_order'); builder.add_edge('select_order','fetch_order')
    builder.add_conditional_edges('fetch_order',lambda s:'expand' if s.get('order_snapshot') else END)
    builder.add_conditional_edges('expand',lambda s:'policy_retrieval' if s.get('expanded_queries') else END)
    builder.add_edge('policy_retrieval',END)
    return builder.compile()
