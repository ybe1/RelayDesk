"""Guarded execution of one model selected tool."""

import asyncio
import json
import logging

from langchain_core.messages import ToolMessage
from pydantic import ValidationError

LOGGER = logging.getLogger(__name__)
READ_ONLY = frozenset({"query_order", "query_product", "query_logistics", "query_faq"})


class ToolExecutor:
    def __init__(self, tools, timeout_seconds: float | None = None, backoff_seconds: float = 0.1):
        self.registry = {item.name: item for item in tools}
        self.timeout_seconds = timeout_seconds
        self.backoff_seconds = backoff_seconds

    async def execute(self, tool_call: dict) -> ToolMessage:
        name = tool_call.get("name", "")
        call_id = tool_call.get("id", "")
        selected = self.registry.get(name)
        if selected is None:
            return self._message(call_id, {"error": {"code": "unknown_tool", "message": "未知工具"}})
        try:
            args = selected.args_schema.model_validate(tool_call.get("args", {})).model_dump()
        except (ValidationError, TypeError, ValueError):
            return self._message(call_id, {"error": {"code": "invalid_arguments", "message": "工具参数不正确"}})

        attempts = 3 if name in READ_ONLY else 1
        for attempt in range(attempts):
            try:
                timeout = self.timeout_seconds if self.timeout_seconds is not None else (120 if name == "query_faq" else 5)
                result = await asyncio.wait_for(selected.ainvoke(args), timeout=timeout)
                return self._message(call_id, {"result": result})
            except Exception as exc:
                LOGGER.warning("Tool %s failed on attempt %s", name, attempt + 1, exc_info=exc)
                if attempt + 1 == attempts or not isinstance(exc, (TimeoutError, ConnectionError, OSError)):
                    code = "timeout" if isinstance(exc, TimeoutError) else "execution_failed"
                    return self._message(call_id, {"error": {"code": code, "message": "工具执行失败"}})
                await asyncio.sleep(self.backoff_seconds * (attempt + 1))
        raise AssertionError("unreachable")

    @staticmethod
    def _message(call_id: str, payload: dict) -> ToolMessage:
        content = json.dumps(payload, ensure_ascii=False)
        if len(content.encode("utf-8")) > 60000:
            raise ValueError("serialized ToolMessage exceeds 60000 UTF-8 bytes")
        return ToolMessage(content=content, tool_call_id=call_id)
