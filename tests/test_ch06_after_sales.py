from tests.ch06_fakes import JSONSequenceModel
from tests.test_ch05_gate import result


async def test_after_sales_calls_order_before_policy():
    from app.workflow.after_sales import build_after_sales_graph
    from app.workflow.contracts import TurnContext,AgentDependencies
    from app.workflow.budget import TurnBudget
    from app.tools.business import build_tools
    from app.services.tool_executor import ToolExecutor
    saved=[]
    async def save(message): saved.append(message)
    async def retrieve(question,queries,budget):
        assert any(m.type=='tool' for m in saved)
        return result()
    tools=build_tools(lambda:None,1)
    context=TurnContext(AgentDependencies(JSONSequenceModel([{'queries':['订单1001能退吗']}]),tools,ToolExecutor(tools)),
        TurnBudget(clock=lambda:0),save,retrieve_queries=retrieve)
    state=await build_after_sales_graph().ainvoke({'question':'订单1001能退吗','intent':'退款退货',
        'order_id':'1001','turn_id':'t','messages':[],'node_trace':[]},context=context)
    assert state['node_trace'].index('fetch_order')<state['node_trace'].index('policy_retrieval')
    assert context.budget.tool_steps==1 and state['order_snapshot']['order_id']=='1001'


def test_offer_requires_refund_intent_eligible_and_strong_cited_evidence():
    from app.workflow.after_sales import refund_offer_defaults
    from app.workflow.order_context import demo_order
    state={'intent':'退款退货','gate_allowed':True,'order_id':'1001','order_snapshot':demo_order('1001'),
        'refund_assessment':{'status':'eligible','reason':'政策允许','evidence_numbers':[1]},
        'evidence':[{'n':1,'chunk_id':1,'answer':'政策'}]}
    assert refund_offer_defaults(state)['order_id']=='1001'
    for update in [{'intent':'售后'},{'gate_allowed':False},{'order_id':'1002'},
                   {'refund_assessment':{'status':'needs_review','reason':'未知','evidence_numbers':[1]}}]:
        assert refund_offer_defaults({**state,**update}) is None


async def test_agent_reuses_only_matching_order_snapshot():
    from app.workflow.agent import build_agent_graph
    from app.workflow.contracts import TurnContext,AgentDependencies
    from app.workflow.budget import TurnBudget
    from app.workflow.order_context import demo_order
    from app.tools.business import build_tools
    from app.services.tool_executor import ToolExecutor
    from langchain_core.messages import HumanMessage
    from tests.test_ch05_agent_graph import GraphModel,tools_decision
    model=GraphModel([tools_decision('query_order'),{'kind':'answer','reason':'够了'}])
    tools=build_tools(lambda:None,1)
    budget=TurnBudget(clock=lambda:0)
    context=TurnContext(AgentDependencies(model,tools,ToolExecutor(tools)),budget)
    state=await build_agent_graph().ainvoke({'messages':[HumanMessage(content='订单1001')],
        'turn_id':'cached','order_snapshot':demo_order('1001')},context=context)
    assert budget.tool_steps==0
    assert len([m for m in state['messages'] if m.type=='tool'])==1
