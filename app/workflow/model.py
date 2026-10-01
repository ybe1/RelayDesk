"""Budget model requests, including failed requests and partial streams."""
import asyncio
from .budget import BudgetExceeded


class BudgetedModel:
    def __init__(self, model, budget, output_cap=1024):
        self.model, self.budget, self.output_cap = model, budget, output_cap

    def _reserve(self, messages, kwargs):
        # UTF-8 bytes deliberately overestimate ordinary text tokens. Include
        # message envelopes/tool requests rather than just visible content.
        inputs = sum(len(str(m.model_dump()).encode("utf-8")) for m in messages) + 128
        cap = min(kwargs.pop("max_tokens", self.output_cap), self.output_cap)
        self.budget.reserve_llm(inputs, cap)
        return cap

    async def ainvoke(self, messages, **kwargs):
        cap = self._reserve(messages, kwargs)
        usage = None
        try:
            async with asyncio.timeout(self.budget.remaining_seconds()):
                result = await self.model.ainvoke(messages, max_tokens=cap, **kwargs)
            usage = getattr(result, "usage_metadata", None)
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
