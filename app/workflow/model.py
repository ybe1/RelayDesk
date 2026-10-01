"""Budget model requests, including failed requests and partial streams."""
import asyncio
import json
from .budget import BudgetExceeded


class BudgetedModel:
    def __init__(self, model, budget, output_cap=1024):
        self.model, self.budget, self.output_cap = model, budget, output_cap

    def with_structured_output(self, schema, **kwargs):
        from langchain_core.language_models.chat_models import BaseChatModel
        if isinstance(self.model, BaseChatModel):
            kwargs["include_raw"] = True
        result = BudgetedModel(self.model.with_structured_output(schema, **kwargs), self.budget, self.output_cap)
        result.structured = True
        return result

    def _reserve(self, messages, kwargs):
        # UTF-8 bytes deliberately overestimate ordinary text tokens. Include
        # message envelopes/tool requests rather than just visible content.
        inputs = sum(len(str(m.model_dump()).encode("utf-8")) for m in messages) + 128
        bound_kwargs = getattr(self.model, "kwargs", {})
        if isinstance(bound_kwargs, dict) and bound_kwargs.get("tools"):
            inputs += len(json.dumps(bound_kwargs["tools"], ensure_ascii=False).encode("utf-8"))
        cap = min(kwargs.pop("max_tokens", self.output_cap), self.output_cap)
        self.budget.reserve_llm(inputs, cap)
        return cap

    async def ainvoke(self, messages, **kwargs):
        cap = self._reserve(messages, kwargs)
        usage = None
        try:
            async with asyncio.timeout(self.budget.remaining_seconds()):
                result = await self.model.ainvoke(messages, max_tokens=cap, **kwargs)
            raw = result.get("raw") if isinstance(result, dict) else None
            usage = getattr(raw or result, "usage_metadata", None)
            if getattr(self, "structured", False) and isinstance(result, dict) and "raw" in result:
                if result.get("parsing_error"):
                    raise ValueError("invalid structured model output") from result["parsing_error"]
                return result["parsed"]
            return result
        except TimeoutError as exc:
            raise BudgetExceeded("deadline") from exc
        finally:
            self.budget.settle_llm(usage)

    async def astream(self, messages, **kwargs):
        cap = self._reserve(messages, kwargs)
        usage = None
        try:
            async with asyncio.timeout(self.budget.remaining_seconds()):
                async for chunk in self.model.astream(messages, max_tokens=cap, **kwargs):
                    usage = getattr(chunk, "usage_metadata", None) or usage
                    yield chunk
        except TimeoutError as exc:
            raise BudgetExceeded("deadline") from exc
        finally:
            self.budget.settle_llm(usage)
