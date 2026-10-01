from datetime import datetime
from secrets import token_hex

from sqlalchemy.orm import Session

from app.db.models import Conversation, Ticket


def create_ticket(session: Session, conversation_id: int, description: str, ticket_type: str) -> Ticket:
    if ticket_type not in {"售后", "投诉", "咨询"}:
        raise ValueError("invalid ticket type")
    conversation = session.get(Conversation, conversation_id)
    if conversation is None:
        raise LookupError("conversation not found")
    ticket = Ticket(ticket_no=f"T{datetime.now():%Y%m%d}{token_hex(5).upper()}", conversation_id=conversation_id, description=description, ticket_type=ticket_type, status="待处理")
    session.add(ticket)
    session.flush()
    return ticket
