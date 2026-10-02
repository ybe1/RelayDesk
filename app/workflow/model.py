"""Budget model requests, including failed requests and partial streams."""
import asyncio
import json
from app.context.tokens import estimate_messages,estimate_text
from .budget import BudgetExceeded


class BudgetedModel:
    def __init__(self, model, budget, output_cap=1024):
        self.model, self.budget, self.output_cap = model, budget, output_cap
        self.context_logger=None

    def with_structured_output(self, schema, **kwargs):
        from langchain_core.language_models.chat_models import BaseChatModel
        if isinstance(self.model, BaseChatModel):
            kwargs["include_raw"] = True
        result = BudgetedModel(self.model.with_structured_output(schema, **kwargs), self.budget, self.output_cap)
        result.structured = True
        result.schema_overhead = estimate_text(json.dumps(schema.model_json_schema(), ensure_ascii=False)) if hasattr(schema, "model_json_schema") else 0
        return result

    def _reserve(self, messages, kwargs):
        # UTF-8 bytes deliberately overestimate ordinary text tokens. Include
        # message envelopes/tool requests rather than just visible content.
        inputs = estimate_messages(messages)
        inputs += getattr(self, "schema_overhead", 0)
        bound_kwargs = getattr(self.model, "kwargs", {})
        if isinstance(bound_kwargs, dict) and bound_kwargs.get("tools"):
            inputs = estimate_messages(messages,bound_kwargs['tools'])+getattr(self,'schema_overhead',0)
        cap = min(kwargs.pop("max_tokens", self.output_cap), self.output_cap)
        # The existing approximate history limit can retain a large old tool
        # result. Fit complete prior user turns to this request's actual
        # conservative preflight; never split a tool request/result pair or
        # remove the current turn/system instructions.
        cfg=getattr(self.budget,'context_config',None)
        while cfg is None and self.budget.tokens + inputs + cap > self.budget.limits.tokens:
            starts = [i for i, message in enumerate(messages) if message.type == "human"]
            if len(starts) < 2:
                break
            first, next_turn = starts[:2]
            removed = messages[first:next_turn]
            inputs -= estimate_messages(removed)
            del messages[first:next_turn]
        if cfg and inputs+cap+max(450,(cfg.window+49)//50)>cfg.window:
            raise BudgetExceeded('上下文预算不足')
        if self.context_logger:
            self.context_logger(messages,bound_kwargs.get('tools',[]) if isinstance(bound_kwargs,dict) else [])
        self.budget.reserve_llm(inputs, cap)
        return cap

    async def ainvoke(self, messages, **kwargs):
        messages = list(messages)
        cap = self._reserve(messages, kwargs)
        usage = None
        try:
            async with asyncio.timeout(self.budget.remaining_seconds()):
                result = await self.model.ainvoke(messages, max_tokens=cap, **kwargs)
            raw = result.get("raw") if isinstance(result, dict) else None
            usage = getattr(raw or result, "usage_metadata", None)
            self._log_usage(messages,usage)
            if getattr(self, "structured", False) and isinstance(result, dict) and "raw" in result:
                if result.get("parsing_error"):
                    raise ValueError("invalid structured model output") from result["parsing_error"]
                return result["parsed"]
            return result
        except TimeoutError as exc:
            raise BudgetExceeded("deadline") from exc
        finally:
            self.budget.settle_llm(usage)

    def _log_usage(self,messages,usage):
        import logging
        logging.getLogger('app.context').info('context_usage estimated_input=%s actual=%s',estimate_messages(messages),usage)

    async def astream(self, messages, **kwargs):
        messages = list(messages)
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
            self._log_usage(messages,usage)
            self.budget.settle_llm(usage)
