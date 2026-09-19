from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64, description="匿名用户 ID")
    conversation_id: int | None = Field(default=None, gt=0, description="服务端生成的会话 ID")
    message: str = Field(min_length=1, description="用户本轮消息")
