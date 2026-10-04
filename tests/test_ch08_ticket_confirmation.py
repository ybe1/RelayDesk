import uuid
import pytest
from sqlalchemy import delete,select,func
from app.db.session import SessionLocal
from app.db.models import Conversation,ChatActionOffer,Ticket
from app.tooling.contracts import CallContext
from app.tooling.permissions import issue_write_permit


@pytest.fixture
def ticket_flow():
    from app.services.ticket_confirmation import TicketConfirmationService,execute_ticket_once
    user='ch08-ticket-'+uuid.uuid4().hex
    with SessionLocal() as session:
        conv=Conversation(user_id=user);session.add(conv);session.commit();cid=conv.id
    ctx=CallContext(conversation_id=cid,user_id=user,turn_id=uuid.uuid4().hex,session_factory=SessionLocal,ticket_request=True)
    call={'name':'create_ticket','args':{'description':'订单1001包裹显示签收但我未收到','ticket_type':'售后'},'id':'call-'+uuid.uuid4().hex}
    service=TicketConfirmationService(SessionLocal)
    preview=service.prepare(ctx,call,'definition-v1')
    try: yield service,ctx,call,preview,execute_ticket_once
    finally:
        with SessionLocal() as session:
            session.execute(delete(ChatActionOffer).where(ChatActionOffer.conversation_id==cid))
            session.execute(delete(Ticket).where(Ticket.conversation_id==cid))
            session.execute(delete(Conversation).where(Conversation.id==cid));session.commit()


def authorize(flow):
    service,ctx,call,preview,_=flow
    service.confirm(ctx.user_id,ctx.conversation_id,preview['confirmation_id'])
    ctx.confirmation=issue_write_permit(ctx,call,'definition-v1',preview['confirmation_id'])


def test_duplicate_confirmation_returns_same_ticket(ticket_flow):
    service,ctx,call,preview,execute=ticket_flow
    authorize(ticket_flow)
    first=execute(ctx,call['args'])
    service.confirm(ctx.user_id,ctx.conversation_id,preview['confirmation_id'])
    second=execute(ctx,call['args'])
    assert first['ticket_no']==second['ticket_no']
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(Ticket).where(Ticket.conversation_id==ctx.conversation_id))==1


def test_cancel_and_cross_user_cannot_create(ticket_flow):
    service,ctx,call,preview,execute=ticket_flow
    with pytest.raises(PermissionError): service.confirm('someone-else',ctx.conversation_id,preview['confirmation_id'])
    service.cancel(ctx.user_id,ctx.conversation_id,preview['confirmation_id'])
    ctx.confirmation=issue_write_permit(ctx,call,'definition-v1',preview['confirmation_id'])
    with pytest.raises(PermissionError): execute(ctx,call['args'])
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(Ticket).where(Ticket.conversation_id==ctx.conversation_id))==0


def test_changed_parameters_or_definition_cannot_execute(ticket_flow):
    service,ctx,call,preview,execute=ticket_flow
    authorize(ticket_flow)
    with pytest.raises(PermissionError): execute(ctx,{**call['args'],'description':'替换内容'})
    ctx.confirmation=issue_write_permit(ctx,call,'different-definition',preview['confirmation_id'])
    with pytest.raises(PermissionError): execute(ctx,call['args'])


def test_crash_after_claim_never_reissues_write(ticket_flow):
    service,ctx,call,preview,execute=ticket_flow
    authorize(ticket_flow)
    claimed=service.claim(ctx,call)
    assert claimed['state']=='claimed'
    result=execute(ctx,call['args'])
    assert result['error']['code']=='outcome_unknown'
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(Ticket).where(Ticket.conversation_id==ctx.conversation_id))==0


def test_confirmation_cannot_move_to_another_turn(ticket_flow):
    service,ctx,call,preview,execute=ticket_flow
    authorize(ticket_flow)
    ctx.turn_id='different-turn'
    with pytest.raises(PermissionError): execute(ctx,call['args'])


@pytest.mark.asyncio
async def test_uncertain_write_result_survives_engine_formatting(ticket_flow):
    import json
    from app.tooling.builtins import bind_ticket
    from app.tooling.contracts import ToolDefinition
    from app.services.tool_executor import ToolExecutor
    service,ctx,call,preview,_=ticket_flow
    # Prepare another independent turn using the actual registered schema hash.
    ctx.turn_id=uuid.uuid4().hex
    tool=bind_ticket(ctx)
    definition=ToolDefinition.from_tool(tool)
    preview=service.prepare(ctx,call,definition.definition_hash)
    service.confirm(ctx.user_id,ctx.conversation_id,preview['confirmation_id'])
    ctx.confirmation=issue_write_permit(ctx,call,definition.definition_hash,preview['confirmation_id'])
    service.claim(ctx,call)
    message=await ToolExecutor([tool],context=ctx).execute(call)
    assert json.loads(message.content)['error']['code']=='outcome_unknown'


@pytest.mark.asyncio
async def test_registered_ticket_uses_confirmation_and_audits_cancel(ticket_flow):
    import json
    from app.tooling.builtins import bind_builtins
    from app.tooling.contracts import ToolDefinition
    from app.services.tool_executor import ToolExecutor
    service,ctx,call,preview,_=ticket_flow
    tool=next(t for t in bind_builtins(ctx) if t.name=='create_ticket')
    definition=ToolDefinition.from_tool(tool)
    # A valid process capability alone cannot override persisted cancellation.
    service.cancel(ctx.user_id,ctx.conversation_id,preview['confirmation_id'])
    ctx.confirmation=issue_write_permit(ctx,call,definition.definition_hash,preview['confirmation_id'])
    class Audits:
        records=[]
        def write(self, record): self.records.append(record)
    audits=Audits()
    message=await ToolExecutor([tool],context=ctx,audits=audits).execute(call)
    assert json.loads(message.content)['error']['code']=='permission_denied'
    assert audits.records[0]['status']=='权限拒绝'
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(Ticket).where(Ticket.conversation_id==ctx.conversation_id))==0
