from sqlalchemy import select

from app.db.models import Conversation, FAQ, Message, Ticket
from app.db.session import SessionLocal
from app.repositories.conversations import get_or_create_conversation
from app.repositories.faq import search_faq
from app.repositories.messages import append_message, load_messages
from app.repositories.tickets import create_ticket


def test_conversation_owner_and_message_order():
    with SessionLocal() as session:
        conversation = get_or_create_conversation(session, "test-owner")
        cid = conversation.id
        append_message(session, cid, "user", "hi")
        append_message(session, cid, "assistant", tool_calls=[{"id": "call_1", "name": "query_faq", "args": {}}])
        append_message(session, cid, "tool", "result", tool_call_id="call_1")
        rows = load_messages(session, cid)
        assert [row.role for row in rows] == ["user", "assistant", "tool"]
        assert rows[1].tool_calls[0]["id"] == rows[2].tool_call_id
        assert get_or_create_conversation(session, "test-owner", cid).id == cid
        try:
            get_or_create_conversation(session, "other-owner", cid)
        except PermissionError:
            pass
        else:
            raise AssertionError("owner mismatch accepted")
        session.rollback()


def test_faq_literal_like_and_ticket_status_transaction():
    with SessionLocal() as session:
        assert search_faq(session, "退货政策")
        assert search_faq(session, "邮费") == []
        assert search_faq(session, "%") == []
        conversation = get_or_create_conversation(session, "ticket-test")
        ticket = create_ticket(session, conversation.id, "需要人工", "咨询")
        assert ticket.ticket_no.startswith("T")
        assert conversation.status == "已转人工"
        session.rollback()
