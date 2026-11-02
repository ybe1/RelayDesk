"""Database chat persistence and complete-turn history trimming."""

import json
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately


def trim_turns(messages: list, budget: int) -> list:
    """Trim complete user turns so tool requests never lose their results."""
    turns: list[list] = []
    for message in messages:
        if isinstance(message, HumanMessage) or not turns:
            turns.append([])
        turns[-1].append(message)
    kept: list[list] = []
    used = 0
    for turn in reversed(turns):
        size = count_tokens_approximately(turn)
        if kept and used + size > budget:
            break
        kept.append(turn)
        used += size
    return [message for turn in reversed(kept) for message in turn]


class DatabaseChatStore:
    def __init__(self, session_factory: Callable):
        self.session_factory = session_factory

    def start_or_load(self, user_id: str, conversation_id: int | None) -> int:
        from app.repositories.conversations import get_or_create_conversation

        with self.session_factory() as session:
            conversation = get_or_create_conversation(session, user_id, conversation_id)
            session.commit()
            return conversation.id

    def append(self, conversation_id: int, message: Any) -> None:
        from app.repositories.messages import append_message

        role = {"human": "user", "ai": "assistant", "tool": "tool"}[message.type]
        content = message.content if isinstance(message.content, str) else json.dumps(message.content, ensure_ascii=False)
        with self.session_factory() as session:
            append_message(
                session,
                conversation_id,
                role,
                content,
                tool_calls=message.tool_calls if isinstance(message, AIMessage) and message.tool_calls else None,
                tool_call_id=message.tool_call_id if isinstance(message, ToolMessage) else None,
            )
            session.commit()

    def load_history(self, conversation_id: int) -> list:
        from app.repositories.messages import load_messages

        with self.session_factory() as session:
            rows = load_messages(session, conversation_id)
            history = []
            for row in rows:
                metadata={'db_id':row.id}
                stable_id=row.message_key or f'legacy:{conversation_id}:{row.id}'
                if row.role == "user":
                    history.append(HumanMessage(content=row.content or "",id=stable_id,additional_kwargs=metadata))
                elif row.role == "assistant":
                    history.append(AIMessage(content=row.content or "", tool_calls=row.tool_calls or [],id=stable_id,additional_kwargs=metadata))
                else:
                    history.append(ToolMessage(content=row.content or "", tool_call_id=row.tool_call_id,id=stable_id,additional_kwargs=metadata))
            return history
