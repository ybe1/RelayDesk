from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from app.db.models import Conversation, Message
from app.db.session import SessionLocal
from app.flywheel.pool import GapEvent, record_gap


router = APIRouter()


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    user_id: str = Field(min_length=1, max_length=64)
    conversation_id: int = Field(gt=0)
    assistant_message_id: int = Field(gt=0)
    helpful: bool


def get_feedback_sessions():
    return SessionLocal


def _owned_turn(factory, payload: FeedbackRequest) -> tuple[str, str]:
    with factory() as session:
        conversation = session.get(Conversation, payload.conversation_id)
        if conversation is None:
            raise LookupError("conversation")
        if conversation.user_id != payload.user_id:
            raise PermissionError("conversation")
        assistant = session.get(Message, payload.assistant_message_id)
        if assistant is None or assistant.conversation_id != payload.conversation_id:
            raise LookupError("message")
        if assistant.role != "assistant" or not assistant.message_key or not assistant.message_key.endswith(":final"):
            raise ValueError("message is not a final assistant answer")
        turn_id = assistant.message_key[:-len(":final")]
        user = session.scalar(select(Message).where(
            Message.conversation_id == payload.conversation_id,
            Message.message_key == f"{turn_id}:user",
            Message.role == "user",
        ))
        if user is None:
            raise RuntimeError("turn user message is unavailable")
        return turn_id, user.content or ""


@router.post("/api/chat/feedback")
async def chat_feedback(request: Request, payload: FeedbackRequest):
    if payload.helpful:
        return {"accepted": True, "low_confidence_id": None}
    factory = getattr(request.app.state, "feedback_session_factory", None) or get_feedback_sessions()
    try:
        turn_id, raw_question = await asyncio.to_thread(_owned_turn, factory, payload)
    except PermissionError as exc:
        raise HTTPException(403, "会话不属于当前用户") from exc
    except LookupError as exc:
        raise HTTPException(404, "会话或回答不存在") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc

    service = getattr(request.app.state, "workflow_service", None)
    if service is None or not getattr(service.graph, "checkpointer", None):
        raise HTTPException(409, "会话检查点暂不可用")
    snapshot = await service.graph.aget_state(service._config(payload.conversation_id))
    final = next((message for message in snapshot.values.get("messages", [])
                  if message.id == f"{turn_id}:final"), None)
    if final is None or "evidence_snapshot" not in final.additional_kwargs:
        raise HTTPException(409, "所选回答的证据快照已失效")
    evidence = final.additional_kwargs.get("evidence_snapshot")
    chunks = None if evidence is None else list(evidence.get("hits", []))[:3]
    event = GapEvent(
        conversation_id=payload.conversation_id,
        raw_question=raw_question,
        source="user_feedback",
        reason="用户反馈该回答未解决问题",
        retrieved_chunks=chunks,
        event_key=f"feedback:{payload.conversation_id}:{payload.assistant_message_id}",
    )
    row_id = await asyncio.to_thread(record_gap, event, factory)
    return {"accepted": True, "low_confidence_id": row_id}
