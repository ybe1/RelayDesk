"""One selection call, at most one tool execution, then a streamed final answer."""

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

from app.config import settings
from app.core.prompts import CUSTOMER_SERVICE_PROMPT
from app.schemas.chat import ChatRequest


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
                if row.role == "user":
                    history.append(HumanMessage(content=row.content or ""))
                elif row.role == "assistant":
                    history.append(AIMessage(content=row.content or "", tool_calls=row.tool_calls or []))
                else:
                    history.append(ToolMessage(content=row.content or "", tool_call_id=row.tool_call_id))
            return history


class ChatService:
    def __init__(self, model, store, tools_factory, executor_factory):
        self.model = model
        self.store = store
        self.tools_factory = tools_factory
        self.executor_factory = executor_factory

    async def events(self, req: ChatRequest) -> AsyncIterator[tuple[str, dict | None]]:
        conversation_id = self.store.start_or_load(req.user_id, req.conversation_id)
        self.store.append(conversation_id, HumanMessage(content=req.message))
        yield "conversation", {"conversation_id": conversation_id}

        history = trim_turns(self.store.load_history(conversation_id), settings.token_budget)
        prompt_messages = CUSTOMER_SERVICE_PROMPT.format_messages(history=history)
        tools = self.tools_factory(conversation_id)
        selection = await self.model.bind_tools(tools, tool_choice="auto").ainvoke(prompt_messages)
        final_messages = list(prompt_messages)

        if selection.tool_calls:
            request_message = AIMessage(content=selection.content or "", tool_calls=selection.tool_calls)
            self.store.append(conversation_id, request_message)
            final_messages.append(request_message)
            pending = {call["id"]: call for call in selection.tool_calls}
            try:
                executor = self.executor_factory(tools)
                for index, call in enumerate(selection.tool_calls):
                    if index == 0:
                        yield "tool_status", {"tool_name": call["name"], "phase": "running", "message": "正在查询"}
                        result = await executor.execute(call)
                    else:
                        result = ToolMessage(
                            content=json.dumps({"error": {"code": "single_tool_limit", "message": "本轮未执行"}}, ensure_ascii=False),
                            tool_call_id=call["id"],
                        )
                    self.store.append(conversation_id, result)
                    pending.pop(call["id"])
                    final_messages.append(result)
                    if index == 0:
                        payload = json.loads(result.content)
                        phase = "error" if "error" in payload else "success"
                        yield "tool_status", {"tool_name": call["name"], "phase": phase, "message": "查询完成" if phase == "success" else "查询失败"}
            finally:
                for call in pending.values():
                    self.store.append(
                        conversation_id,
                        ToolMessage(
                            content=json.dumps({"error": {"code": "interrupted", "message": "工具执行被中断，结果未知"}}, ensure_ascii=False),
                            tool_call_id=call["id"],
                        ),
                    )

        chunks: list[str] = []
        async for chunk in self.model.astream(final_messages):
            piece = chunk.content if isinstance(chunk.content, str) else ""
            if piece:
                chunks.append(piece)
                yield "delta", {"delta": piece}
        self.store.append(conversation_id, AIMessage(content="".join(chunks)))
        yield "done", None
