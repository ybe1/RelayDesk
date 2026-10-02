import asyncio
from typing import Literal
from fastapi import APIRouter,HTTPException
from pydantic import BaseModel,ConfigDict,Field
from app.db.session import SessionLocal
from app.services.refunds import confirm_refund

router = APIRouter()


class RefundConfirmation(BaseModel):
    model_config = ConfigDict(extra='forbid',strict=True,str_strip_whitespace=True)
    user_id: str = Field(min_length=1,max_length=64)
    conversation_id: int = Field(gt=0)
    offer_id: str = Field(min_length=1,max_length=64)
    reason: Literal['不喜欢/不需要','商品质量问题','商品与描述不符','错发/漏发','其他']


@router.post('/api/chat/actions/refund')
async def confirm(req:RefundConfirmation):
    try:
        return await asyncio.to_thread(confirm_refund,SessionLocal,req.user_id,req.conversation_id,req.offer_id,req.reason)
    except PermissionError as exc: raise HTTPException(403,'会话或选项不属于当前用户') from exc
    except LookupError as exc: raise HTTPException(404,'会话或选项不存在') from exc
    except ValueError as exc: raise HTTPException(409,str(exc)) from exc
