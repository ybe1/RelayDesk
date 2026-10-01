from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.db.models import Message


def append_message(session: Session, conversation_id: int, role: str, content: str | None = None, tool_calls: list[dict] | None = None, tool_call_id: str | None = None) -> Message:
    message = Message(conversation_id=conversation_id, role=role, content=content, tool_calls=tool_calls, tool_call_id=tool_call_id)
    session.add(message)
    session.flush()
    return message


def load_messages(session: Session, conversation_id: int) -> list[Message]:
    return list(session.scalars(select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id)))


def append_once(session: Session, conversation_id: int, message_key: str, message) -> Message:
    """Append a stable message once; replays must match the original payload."""
    if not message_key or len(message_key) > 128:
        raise ValueError("invalid message_key")
    role = {"human": "user", "ai": "assistant", "tool": "tool"}.get(message.type)
    if role is None:
        raise ValueError("unsupported message role")
    payload = dict(conversation_id=conversation_id, role=role, content=message.content,
                   tool_calls=getattr(message, "tool_calls", None) or None,
                   tool_call_id=getattr(message, "tool_call_id", None))
    # Locking reads see current committed rows even under MySQL REPEATABLE READ.
    query = select(Message).where(Message.message_key == message_key).with_for_update()
    existing = session.scalar(query)
    if existing is None:
        try:
            with session.begin_nested():
                existing = Message(message_key=message_key, **payload)
                session.add(existing)
                session.flush()
            return existing
        except IntegrityError:
            existing = session.scalar(query)
            if existing is None:
                raise
    if any(getattr(existing, key) != value for key, value in payload.items()):
        raise ValueError("message_key payload mismatch")
    return existing


def close_pending_tools(session: Session, conversation_id: int) -> int:
    """Close persisted requests after interruption; never execute them again.

    Caller must hold the conversation lock and call before starting a new turn.
    """
    import json
    import hashlib
    from langchain_core.messages import ToolMessage
    rows = load_messages(session, conversation_id)
    answered = {row.tool_call_id for row in rows if row.role == "tool"}
    count = 0
    for row in rows:
        if row.role != "assistant":
            continue
        for call in row.tool_calls or []:
            call_id = call.get("id")
            if not call_id or call_id in answered:
                continue
            key = f"recovery:{conversation_id}:{row.id}:" + hashlib.sha256(call_id.encode()).hexdigest()[:32]
            append_once(session, conversation_id, key, ToolMessage(tool_call_id=call_id,
                content=json.dumps({"error": {"code": "interrupted", "message": "上次工具调用被中断，未恢复执行"}}, ensure_ascii=False)))
            answered.add(call_id)
            count += 1
    return count
