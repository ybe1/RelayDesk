from tests.ch06_fakes import JSONSequenceModel
from tests.test_ch05_gate import result


async def test_general_policy_retrieves_without_order_or_expansion():
    from app.workflow.graph import build_workflow,make_turn_input
    from app.workflow.contracts import TurnContext,AgentDependencies
    from app.workflow.budget import TurnBudget
    from app.tools.business import build_tools
    from app.services.tool_executor import ToolExecutor
    from langchain_core.messages import HumanMessage
    model=JSONSequenceModel([
        {'question':'退货政策是什么','reference_status':'clear','scope':'general','order_id':None,'order_source':'none'},
        {'intent':'退款退货','confidence':0.95},
        {'kind':'answer','reason':'证据足够','tool_calls':[],'actions':[]}])
    async def retrieve(question,budget): return result()
    tools=build_tools(lambda:None,1)
    context=TurnContext(AgentDependencies(model,tools,ToolExecutor(tools)),TurnBudget(clock=lambda:0),retrieve=retrieve)
    state=await build_workflow().ainvoke(make_turn_input('退货政策是什么',1,'u','t',[HumanMessage(content='退货政策是什么')]),context=context)
    assert state['resolution']['scope']=='general'
    assert 'retrieval' in state['node_trace'] and 'fetch_order' not in state['node_trace']
    assert 'expand' not in state['node_trace'] and state['stop_reason']=='answered'


def test_reset_clears_previous_turn_order_and_assessment():
    from app.workflow.graph import make_turn_input
    state=make_turn_input('新问题',1,'u','new',[])
    assert state['order_id'] is None and state['refund_assessment']=={} and state['selection']=={}


async def test_sqlite_interrupt_resumes_without_repeating_initial_models(tmp_path):
    from app.workflow.runtime import checkpoint_runtime
    from app.workflow.graph import build_workflow,make_turn_input
    from app.workflow.contracts import TurnContext,AgentDependencies
    from app.workflow.budget import TurnBudget
    from app.tools.business import build_tools
    from app.services.tool_executor import ToolExecutor
    from langgraph.types import Command
    from langchain_core.messages import HumanMessage
    model=JSONSequenceModel([
        {'question':'这个能退吗','reference_status':'ambiguous','scope':'order_specific','order_id':None,'order_source':'none'},
        {'intent':'退款退货','confidence':0.95},
        {'queries':['这个能退吗','退货政策']},
        {'kind':'answer','reason':'尚需核实','actions':[],'refund_assessment':{'status':'needs_review','reason':'未知商品状态','evidence_numbers':[1]}}])
    async def retrieve(question,queries,budget): return result()
    tools=build_tools(lambda:None,1)
    context=TurnContext(AgentDependencies(model,tools,ToolExecutor(tools)),TurnBudget(clock=lambda:0),retrieve_queries=retrieve)
    config={'configurable':{'thread_id':'selection-test'}}
    async with checkpoint_runtime(tmp_path/'select.sqlite') as saver:
        graph=build_workflow(checkpointer=saver)
        first=await graph.ainvoke(make_turn_input('这个能退吗',1,'u','t',[HumanMessage(content='这个能退吗')]),config,context=context)
        selection=first['__interrupt__'][0].value
        assert len(model.requests)==2 and len(selection['options'])==3
        last=await graph.ainvoke(Command(resume={'selection_id':selection['selection_id'],'order_id':'1001'}),config,context=context)
        assert len(model.requests)==5 and last['node_trace'].count('prepare_selection')==1
        assert last['order_id']=='1001' and last['refund_assessment']['status']=='needs_review'
