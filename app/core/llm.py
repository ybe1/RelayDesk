from langchain_openai import ChatOpenAI

from app.config import settings

def get_chat_model(streaming: bool = False, disable_thinking: bool = False,
                   temperature: float = 0.3, max_retries: int | None = None,
                   stream_usage: bool | None = None, model_name: str | None = None,
                   http_async_client=None) -> ChatOpenAI:
    """直连聊天上游。地址、模型名、密钥全从 .env 来,换上游不动这里。"""
    kwargs: dict = {
        "model": model_name or settings.chat_model,
        "base_url": settings.chat_base_url,
        "api_key": settings.chat_api_key,
        "streaming": streaming,
        "temperature": temperature,
    }
    if disable_thinking:
        # deepseek-v4-pro 是推理模型,思考模式下不支持 tool_choice;结构化提取需关闭
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    if stream_usage is not None:
        kwargs["stream_usage"] = stream_usage
    if http_async_client is not None:
        kwargs['http_async_client']=http_async_client
    return ChatOpenAI(**kwargs)
