from langchain_openai import ChatOpenAI

from app.config import settings

def get_chat_model(streaming: bool = False, disable_thinking: bool = False) -> ChatOpenAI:
    """直连聊天上游。地址、模型名、密钥全从 .env 来,换上游不动这里。"""
    kwargs: dict = {
        "model": settings.chat_model,
        "base_url": settings.chat_base_url,
        "api_key": settings.chat_api_key,
        "streaming": streaming,
        "temperature": 0.3,
    }
    if disable_thinking:
        # deepseek-v4-pro 是推理模型,思考模式下不支持 tool_choice;结构化提取需关闭
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    return ChatOpenAI(**kwargs)
