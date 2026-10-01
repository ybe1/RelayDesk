import asyncio
from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from app.db.session import SessionLocal
from app.services.chat_actions import confirm_ticket

router = APIRouter()


class TicketConfirmation(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    user_id: str = Field(min_length=1, max_length=64)
    conversation_id: int = Field(gt=0, strict=True)
    offer_id: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=4000)
    ticket_type: Literal["售后", "投诉", "咨询"]


@router.post("/api/chat/actions/ticket")
async def confirm(req: TicketConfirmation):
    try:
        return await asyncio.to_thread(confirm_ticket, SessionLocal, req.user_id, req.conversation_id,
                                       req.offer_id, req.description, req.ticket_type)
    except PermissionError as exc:
        raise HTTPException(403, "会话或选项不属于当前用户") from exc
    except LookupError as exc:
        raise HTTPException(404, "会话或选项不存在") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
