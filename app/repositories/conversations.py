from sqlalchemy.orm import Session

from app.db.models import Conversation


def get_or_create_conversation(session: Session, user_id: str, conversation_id: int | None = None) -> Conversation:
    if conversation_id is None:
        conversation = Conversation(user_id=user_id)
        session.add(conversation)
        session.flush()
        return conversation
    conversation = session.get(Conversation, conversation_id)
    if conversation is None:
        raise LookupError("conversation not found")
    if conversation.user_id != user_id:
        raise PermissionError("conversation belongs to another user")
    return conversation
