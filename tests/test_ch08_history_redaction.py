from uuid import uuid4

from sqlalchemy import delete

from app.db.models import ChatActionOffer, Conversation, Message
from app.db.session import SessionLocal
from app.repositories.conversations import owned_history


def test_ticket_confirmation_internal_state_is_not_exposed_in_history():
    user = "ch08-history-" + uuid4().hex
    turn = uuid4().hex
    offer_id = "offer-" + uuid4().hex
    with SessionLocal() as session:
        conversation = Conversation(user_id=user)
        session.add(conversation)
        session.flush()
        cid = conversation.id
        session.add(Message(conversation_id=cid, role="assistant", content="请确认工单", message_key=f"{turn}:final"))
        session.add(ChatActionOffer(
            offer_id=offer_id,
            conversation_id=cid,
            turn_id=turn,
            action="ticket",
            defaults={"description": "包裹未收到", "ticket_type": "售后", "_ch08": {
                "state": "prepared", "tool_call_id": "private-call", "definition_hash": "private-hash",
                "claim_token": "private-token", "arguments": {"secret": "private-value"},
            }},
        ))
        session.commit()

    try:
        with SessionLocal() as session:
            result = owned_history(session, user, cid)
        action = result["messages"][0]["actions"][0]
        assert action["defaults"] == {"description": "包裹未收到", "ticket_type": "售后"}
        assert "private-token" not in str(result)
        assert "private-value" not in str(result)
    finally:
        with SessionLocal() as session:
            session.execute(delete(ChatActionOffer).where(ChatActionOffer.conversation_id == cid))
            session.execute(delete(Message).where(Message.conversation_id == cid))
            session.execute(delete(Conversation).where(Conversation.id == cid))
            session.commit()
