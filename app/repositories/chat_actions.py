from uuid import uuid4
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from app.db.models import ChatActionOffer


def issue_offer(session, conversation_id, turn_id, action, defaults):
    if action not in {"handoff", "ticket"}:
        raise ValueError("unknown action")
    query = select(ChatActionOffer).where(ChatActionOffer.conversation_id == conversation_id,
                ChatActionOffer.turn_id == turn_id, ChatActionOffer.action == action).with_for_update()
    existing = session.scalar(query)
    if existing is not None:
        return existing
    try:
        with session.begin_nested():
            offer = ChatActionOffer(offer_id=uuid4().hex, conversation_id=conversation_id,
                                    turn_id=turn_id, action=action, defaults=defaults)
            session.add(offer)
            session.flush()
        return offer
    except IntegrityError:
        existing = session.scalar(query)
        if existing is None:
            raise
        return existing
