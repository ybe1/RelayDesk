import pytest
from langchain_core.messages import HumanMessage
from app.db.session import SessionLocal
from app.repositories.conversations import get_or_create_conversation
from app.repositories.tickets import create_ticket


def test_ticket_does_not_transfer_conversation():
    with SessionLocal() as session:
        conversation = get_or_create_conversation(session, "ch05-ticket")
        ticket = create_ticket(session, conversation.id, "投诉演示", "投诉")
        assert ticket.conversation_id == conversation.id
        assert conversation.status == "进行中"
        session.rollback()


def test_interrupted_tool_requests_are_closed_once():
    from langchain_core.messages import AIMessage
    from app.repositories.messages import append_once, close_pending_tools, load_messages
    import json
    with SessionLocal() as session:
        conversation = get_or_create_conversation(session, "ch05-interrupted")
        append_once(session, conversation.id, "test-ch05-interrupted:assistant", AIMessage(content="", tool_calls=[
            {"name": "query_order", "args": {"order_id": "1001"}, "id": "interrupted-call"}]))
        assert close_pending_tools(session, conversation.id) == 1
        assert close_pending_tools(session, conversation.id) == 0
        rows = load_messages(session, conversation.id)
        assert rows[-1].tool_call_id == "interrupted-call"
        assert json.loads(rows[-1].content)["error"]["code"] == "interrupted"
        session.rollback()


def test_replayed_message_is_one_row_and_payload_cannot_change():
    from app.repositories.messages import append_once, load_messages
    with SessionLocal() as session:
        conversation = get_or_create_conversation(session, "ch05-message")
        first = append_once(session, conversation.id, "test-ch05-turn:user", HumanMessage(content="你好"))
        second = append_once(session, conversation.id, "test-ch05-turn:user", HumanMessage(content="你好"))
        assert first.id == second.id
        assert len(load_messages(session, conversation.id)) == 1
        with pytest.raises(ValueError, match="message_key"):
            append_once(session, conversation.id, "test-ch05-turn:user", HumanMessage(content="被改写"))
        other = get_or_create_conversation(session, "ch05-other")
        with pytest.raises(ValueError, match="message_key"):
            append_once(session, other.id, "test-ch05-turn:user", HumanMessage(content="你好"))
        session.rollback()
