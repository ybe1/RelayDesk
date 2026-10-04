import pytest
from langchain_core.messages import HumanMessage,AIMessageChunk
from app.context.contracts import HistoryProjection,LayerAnchors,ContextConfig
from app.workflow.contracts import AgentDependencies,TurnContext
from app.workflow.budget import TurnBudget,Limits


@pytest.mark.parametrize('summarized',[False,True])
async def test_recall_uses_session_facts_without_faq_or_business_tools(summarized):
    from app.workflow.graph import build_workflow,make_turn_input
    from app.tools.business import build_tools
    facts='演示订单1001，虚拟联系手机号13800000000，用户希望核实包裹为什么还没到。'
    projection=HistoryProjection(LayerAnchors(),() if summarized else (HumanMessage(content=facts,id='old'),),(),facts if summarized else '',0,())
    class Model:
        def bind_tools(self,tools,**kwargs):
            assert kwargs['tool_choice']=='none'
            return self
        async def ainvoke(self,*args,**kwargs): raise AssertionError('recall must not classify or select tools')
        async def astream(self,messages,**kwargs):
            self.messages=messages
            assert any('13800000000' in str(m.content) for m in messages)
            yield AIMessageChunk(content='最开始的订单号是1001，您留下的虚拟手机号是13800000000，诉求是核实包裹未到。')
    class Executor:
        async def execute(self,*args,**kwargs): raise AssertionError('history recall must not execute business tools')
    async def retrieve(*args): raise AssertionError('session facts are not KB evidence')
    model=Model()
    context=TurnContext(AgentDependencies(model,build_tools(lambda:None,0),Executor()),TurnBudget(Limits(tokens=100000)),
        retrieve=retrieve,history_projection=projection,context_config=ContextConfig())
    question='最开始那个订单号是什么？我留下的联系手机号是什么？当时想让你帮我做什么？'
    state=make_turn_input(question,1,'u','current',[HumanMessage(content=question,id='current:user')])
    result=await build_workflow().ainvoke(state,context=context)
    assert result['stop_reason']=='answered'
    assert 'recall' in result['node_trace'] and 'retrieval' not in result['node_trace']
    assert '13800000000' in result['final_text']
    assert context.budget.tool_steps==0


@pytest.mark.parametrize('question',['刚才那个订单能退吗','之前那个商品的退货政策是什么','订单1001的物流到哪了','我的手机号是多少'])
def test_business_or_identity_without_history_reference_not_intercepted(question):
    from app.workflow.history import is_history_recall
    assert not is_history_recall(question)


def test_recall_flag_does_not_leak_into_next_turn():
    from app.workflow.graph import make_turn_input
    assert make_turn_input('订单1001能退吗',1,'u','new',[])['history_recall'] is False
