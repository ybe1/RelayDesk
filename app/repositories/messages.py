from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Message


def append_message(session: Session, conversation_id: int, role: str, content: str | None = None, tool_calls: list[dict] | None = None, tool_call_id: str | None = None) -> Message:
    message = Message(conversation_id=conversation_id, role=role, content=content, tool_calls=tool_calls, tool_call_id=tool_call_id)
    session.add(message)
    session.flush()
    return message


def load_messages(session: Session, conversation_id: int) -> list[Message]:
    return list(session.scalars(select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id)))
