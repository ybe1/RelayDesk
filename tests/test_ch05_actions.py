from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
import pytest
from sqlalchemy import select, delete
from app.db.session import SessionLocal
from app.db.models import Conversation, Ticket
from app.repositories.conversations import get_or_create_conversation


@pytest.fixture
def action_case():
    from app.repositories.chat_actions import issue_offer
    from app.db.models import ChatActionOffer
    user = "ch05-actions-" + uuid4().hex[:12]
    with SessionLocal() as session:
        conversation = get_or_create_conversation(session, user)
        cid = conversation.id
        ticket = issue_offer(session, cid, "test-turn", "ticket", {"description": "投诉演示", "ticket_type": "投诉"})
        handoff = issue_offer(session, cid, "test-turn", "handoff", {})
        ids = (ticket.offer_id, handoff.offer_id)
        session.commit()
    yield user, cid, ids
    with SessionLocal() as session:
        session.execute(delete(ChatActionOffer).where(ChatActionOffer.conversation_id == cid))
        session.execute(delete(Ticket).where(Ticket.conversation_id == cid))
        session.execute(delete(Conversation).where(Conversation.id == cid))
        session.commit()


def test_ignored_offers_create_no_ticket(action_case):
    user, cid, _ = action_case
    with SessionLocal() as session:
        assert not list(session.scalars(select(Ticket).where(Ticket.conversation_id == cid)))
        assert session.get(Conversation, cid).status == "进行中"


def test_double_confirmation_and_changed_params(action_case):
    from app.services.chat_actions import confirm_ticket
    user, cid, (offer, _) = action_case
    first = confirm_ticket(SessionLocal, user, cid, offer, "投诉演示", "投诉")
    second = confirm_ticket(SessionLocal, user, cid, offer, "投诉演示", "投诉")
    assert first == second
    with pytest.raises(ValueError): confirm_ticket(SessionLocal, user, cid, offer, "改内容", "投诉")
    with SessionLocal() as session:
        assert len(list(session.scalars(select(Ticket).where(Ticket.conversation_id == cid)))) == 1
        assert session.get(Conversation, cid).status == "进行中"
        from app.db.models import ToolAuditLog
        audits=list(session.scalars(select(ToolAuditLog).where(ToolAuditLog.conversation_id==cid)))
        assert len(audits)==1
        assert audits[0].tool_name=='create_ticket' and audits[0].status=='成功'


def test_wrong_owner_and_handoff_offer_are_rejected(action_case):
    from app.services.chat_actions import confirm_ticket
    user, cid, (offer, handoff) = action_case
    with pytest.raises(PermissionError): confirm_ticket(SessionLocal, "someone-else", cid, offer, "投诉", "投诉")
    with pytest.raises(ValueError): confirm_ticket(SessionLocal, user, cid, handoff, "投诉", "投诉")


def test_mysql_concurrent_confirmations_return_same_ticket(action_case):
    from app.services.chat_actions import confirm_ticket
    user, cid, (offer, _) = action_case
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(confirm_ticket, SessionLocal, user, cid, offer, "并发演示", "咨询") for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]
    assert results[0]["ticket_no"] == results[1]["ticket_no"]
