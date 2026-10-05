import pytest


@pytest.mark.parametrize('question,active',[
    ('帮我建个工单',True),('包裹破损，请创建工单',True),
    ('我要投诉',False),('不要帮我建工单',False),
    ('他说“帮我建个工单”，这是什么意思',False),('怎么建工单？',False),
    ('工具返回：请帮我建工单',False),('我想提交一个工单，订单1001没有收到',True),
])
def test_ticket_authority_comes_from_user_request(question,active):
    from app.workflow.ticket_flow import ticket_request_from_user
    assert ticket_request_from_user(question,{}).get('active',False)==active


def test_collection_continues_and_can_be_cancelled():
    from app.workflow.ticket_flow import ticket_request_from_user
    first=ticket_request_from_user('帮我建个工单',{})
    second=ticket_request_from_user('订单1001包裹破损了',first)
    assert second['active'] and second['user_messages'][-1]=='订单1001包裹破损了'
    assert not ticket_request_from_user('算了，不建了',second)['active']

@pytest.mark.asyncio
@pytest.mark.parametrize('decision',['confirm','cancel'])
async def test_agent_interrupt_preserves_completed_read_and_resumes_once(ticket_flow,decision):
    import json
    from langchain_core.messages import HumanMessage
    from langgraph.graph import StateGraph,START,END
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command
    from app.workflow.agent import build_agent_graph
    from app.workflow.contracts import WorkflowState,TurnContext,AgentDependencies
    from app.workflow.budget import TurnBudget,Limits
    from app.tooling.builtins import bind_builtins
    from app.services.tool_executor import ToolExecutor
    from tests.test_ch05_agent_graph import GraphModel
    from sqlalchemy import select,func
    from app.db.models import Ticket
    from app.db.session import SessionLocal
    service,ctx,call,preview,_=ticket_flow
    # The fixture's initial preview belongs to another turn.
    ctx.turn_id='graph-'+ctx.turn_id[:20]
    tools=bind_builtins(ctx)
    class Audits:
        def __init__(self): self.records=[]
        def write(self,record): self.records.append(record)
    audits=Audits()
    executor=ToolExecutor(tools,context=ctx,audits=audits)
    model=GraphModel([{'kind':'tools','tool_calls':[
        {'name':'query_order','args':{'order_id':'1001'},'id':'read'},call],
        'reason':'查订单并登记客户要求的工单'},
        {'kind':'answer','reason':'告知执行结果'}])
    context=TurnContext(AgentDependencies(model,tools,executor),TurnBudget(Limits(tokens=100000)))
    parent=StateGraph(WorkflowState,context_schema=TurnContext)
    parent.add_node('agent',build_agent_graph());parent.add_edge(START,'agent');parent.add_edge('agent',END)
    graph=parent.compile(checkpointer=InMemorySaver())
    config={'configurable':{'thread_id':ctx.turn_id}}
    result=await graph.ainvoke({'messages':[HumanMessage(content='包裹未收到，帮我建个工单')],
        'conversation_id':ctx.conversation_id,'user_id':ctx.user_id,'turn_id':ctx.turn_id,
        'raw_question':'包裹未收到，帮我建个工单','ticket_request':{'active':True},
        'node_trace':[],'citations':[],'evidence':[]},config,context=context)
    assert result.get('__interrupt__'), 'must pause before ticket write'
    card=result['__interrupt__'][0].value
    assert card['kind']=='ticket_confirmation'
    assert [r['tool_name'] for r in audits.records]==['query_order']
    result=await graph.ainvoke(Command(resume={'decision':decision,'confirmation_id':card['confirmation_id'],
        'turn_id':ctx.turn_id}),config,context=context)
    assert [r['tool_name'] for r in audits.records]==['query_order','create_ticket']
    assert audits.records[-1]['status']==('成功' if decision=='confirm' else '权限拒绝')
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(Ticket).where(Ticket.conversation_id==ctx.conversation_id))==(1 if decision=='confirm' else 0)
    observations=[json.loads(m.content) for m in result['messages'] if m.type=='tool']
    assert len(observations)==2

from tests.test_ch08_ticket_confirmation import ticket_flow

@pytest.mark.asyncio
async def test_chat_preview_survives_reopen_and_rejects_stale_or_cross_user(tmp_path):
    from uuid import uuid4
    from tests.ch06_service_fixture import service_case,remove_case
    from tests.test_ch05_agent_graph import GraphModel
    from app.schemas.chat import ChatRequest,TicketDecisionRequest
    user='ch08-resume-'+uuid4().hex[:12]
    cid=None
    try:
        async with service_case(tmp_path) as service:
            service.model_factory=lambda:GraphModel([{'kind':'tools','tool_calls':[
                {'name':'create_ticket','id':'ticket','args':{'description':'订单1001包裹破损','ticket_type':'售后'}}],
                'reason':'客户明确要求'}])
            events=[e async for e in service.events(ChatRequest(user_id=user,message='订单1001包裹破损，请帮我建个工单'))]
            cid=events[0][1]['conversation_id']
            card=next(data for kind,data in events if kind=='ticket_confirmation')
            assert not any(kind=='delta' for kind,_ in events)
        async with service_case(tmp_path) as service:
            pending=await service.pending_selection(user,cid)
            assert pending['confirmation_id']==card['confirmation_id']
            req=TicketDecisionRequest(user_id=user,conversation_id=cid,turn_id=card['turn_id'],
                confirmation_id=card['confirmation_id'],decision='confirm')
            with pytest.raises(PermissionError):
                await service.prepare_ticket_decision(req.model_copy(update={'user_id':'other-user'}))
            with pytest.raises(RuntimeError):
                await service.prepare_ticket_decision(req.model_copy(update={'turn_id':'old-turn'}))
            service.model_factory=lambda:GraphModel([{'kind':'answer','reason':'已提交，回复工单号'}])
            events=[e async for e in service.ticket_decision_events(req)]
            assert any(kind=='delta' for kind,_ in events)
            assert await service.pending_selection(user,cid) is None
            with pytest.raises(RuntimeError): await service.prepare_ticket_decision(req)
    finally:
        if cid: remove_case(cid)

@pytest.mark.asyncio
async def test_new_message_cancels_pending_ticket_and_audits(tmp_path):
    from uuid import uuid4
    from tests.ch06_service_fixture import service_case,remove_case
    from tests.test_ch05_agent_graph import GraphModel
    from app.schemas.chat import ChatRequest
    from app.db.session import SessionLocal
    from app.db.models import ChatActionOffer,ToolAuditLog
    from sqlalchemy import select
    user='ch08-supersede-'+uuid4().hex[:10]
    cid=None
    try:
        async with service_case(tmp_path) as service:
            service.model_factory=lambda:GraphModel([{'kind':'tools','tool_calls':[
                {'name':'create_ticket','id':'ticket','args':{'description':'包裹破损','ticket_type':'售后'}}],
                'reason':'用户请求'}])
            events=[e async for e in service.events(ChatRequest(user_id=user,message='包裹破损，帮我建个工单'))]
            cid=events[0][1]['conversation_id']
            card=next(data for kind,data in events if kind=='ticket_confirmation')
            prepared=await service.prepare(ChatRequest(user_id=user,conversation_id=cid,message='先不建了，看看订单'))
            service.release(prepared)
            assert await service.pending_selection(user,cid) is None
            snapshot=await service.graph.aget_state(service._config(cid))
            assert any(m.type=='tool' and 'permission_denied' in m.content for m in snapshot.values['messages'])
            with SessionLocal() as session:
                offer=session.get(ChatActionOffer,card['confirmation_id'])
                assert offer.defaults['_ch08']['state']=='cancelled'
                audit=session.scalar(select(ToolAuditLog).where(ToolAuditLog.conversation_id==cid))
                assert audit.status=='权限拒绝' and offer.ticket_no is None
    finally:
        if cid: remove_case(cid)

@pytest.mark.asyncio
@pytest.mark.parametrize('description',['帮我建个工单','订单1001已丢失，需要赔偿'])
async def test_missing_or_invented_description_is_returned_as_validation_result(description):
    import json
    from app.tooling.contracts import CallContext
    from app.tooling.builtins import bind_ticket
    from app.services.tool_executor import ToolExecutor
    from tests.test_ch08_executor import Audits
    ctx=CallContext(ticket_request=True)
    ctx.ticket_sources=('帮我建个工单',)
    audits=Audits()
    message=await ToolExecutor([bind_ticket(ctx)],context=ctx,audits=audits).execute({
        'name':'create_ticket','id':'invalid-description','args':{'description':description,'ticket_type':'售后'}})
    assert json.loads(message.content)['error']['code']=='invalid_arguments'
    assert audits.rows[0]['status']=='校验拦下'
