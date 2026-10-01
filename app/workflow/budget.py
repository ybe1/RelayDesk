"""One shared, serializable budget for every model call in a turn."""
from dataclasses import dataclass
from time import monotonic


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class Limits:
    model_calls: int = 6
    tools: int = 5
    tokens: int = 12000
    seconds: float = 120


class TurnBudget:
    def __init__(self, limits=None, clock=monotonic, usage=None):
        self.limits = limits or Limits()
        self.clock = clock
        self.started = clock()
        usage = usage or {}
        self.elapsed = usage.get("elapsed", 0)
        self.model_calls = usage.get("model_calls", 0)
        self.tool_steps = usage.get("tool_steps", 0)
        self.tokens = usage.get("tokens", 0)
        self.estimated = usage.get("estimated", False)
        self.external_usage = {name: dict(value) for name, value in usage.get("external_usage", {}).items()}
        self.reservation = 0

    def remaining_seconds(self):
        return max(0, self.limits.seconds - self.elapsed - (self.clock() - self.started))

    def check_deadline(self):
        if self.remaining_seconds() <= 0:
            raise BudgetExceeded("deadline")

    def reserve_llm(self, input_tokens, output_cap):
        self.check_deadline()
        if self.model_calls >= self.limits.model_calls:
            raise BudgetExceeded("model_calls")
        if self.tokens + input_tokens + output_cap > self.limits.tokens:
            raise BudgetExceeded("tokens")
        if self.reservation:
            raise RuntimeError("unsettled model request")
        self.reservation = input_tokens + output_cap
        self.model_calls += 1

    def settle_llm(self, usage):
        total = usage.get("total_tokens") if isinstance(usage, dict) else None
        if not isinstance(total, int) or isinstance(total, bool) or total < 0:
            total = self.reservation
            self.estimated = True
        self.tokens += total
        self.reservation = 0

    def reserve_tool(self):
        self.check_deadline()
        if self.tool_steps >= self.limits.tools:
            raise BudgetExceeded("tool_steps")
        self.tool_steps += 1

    def snapshot(self):
        return {"model_calls": self.model_calls, "tool_steps": self.tool_steps,
                "tokens": self.tokens + self.reservation, "estimated": self.estimated,
                "elapsed": self.elapsed + self.clock() - self.started,
                "external_usage": {name: dict(value) for name, value in self.external_usage.items()}}

    def record_external(self, name, seconds):
        entry = self.external_usage.setdefault(name, {"calls": 0, "seconds": 0.0})
        entry["calls"] += 1
        entry["seconds"] += max(0, seconds)
