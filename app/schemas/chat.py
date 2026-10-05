from pydantic import BaseModel, Field,ConfigDict
from typing import Literal

class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64, description="匿名用户 ID")
    conversation_id: int | None = Field(default=None, gt=0, description="服务端生成的会话 ID")
    message: str = Field(min_length=1, description="用户本轮消息")


class SelectionRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    user_id:str=Field(min_length=1,max_length=64)
    conversation_id:int=Field(gt=0)
    turn_id:str=Field(min_length=1,max_length=64)
    selection_id:str=Field(min_length=1,max_length=64)
    order_id:str=Field(min_length=1,max_length=64)


class TicketDecisionRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    user_id:str=Field(min_length=1,max_length=64)
    conversation_id:int=Field(gt=0)
    turn_id:str=Field(min_length=1,max_length=64)
    confirmation_id:str=Field(min_length=1,max_length=64)
    decision:Literal['confirm','cancel']
