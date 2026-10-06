from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class IntentUsage(BaseModel):
    intent: str
    requests: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    usage_missing: int
    cost: float | None


def _intent(tags: list[str] | None) -> str:
    values = [tag.removeprefix("intent:") for tag in (tags or []) if tag.startswith("intent:")]
    return values[0] if values else "未分类"


def _tokens(usage: dict[str, int], key: str) -> int:
    return int(usage.get(key, usage.get(f"{key}_tokens", usage.get(f"{key}Tokens", 0))) or 0)


def summarize_usage(start: datetime, end: datetime, client: Any) -> list[IntentUsage]:
    """Aggregate online model-generation leaves without counting parent spans."""
    buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"traces": set(), "input": 0, "output": 0, "total": 0,
                 "missing": 0, "cost": 0.0, "has_cost": False}
    )
    cursor = None
    while True:
        page = client.api.observations.get_many(
            fields="core,basic,usage,trace_context",
            type="GENERATION",
            from_start_time=start,
            to_start_time=end,
            limit=1000,
            cursor=cursor,
        )
        for row in page.data:
            tags = list(getattr(row, "tags", None) or [])
            if getattr(row, "type", None) != "GENERATION" or "online" not in tags:
                continue
            bucket = buckets[_intent(tags)]
            trace_id = getattr(row, "trace_id", None)
            if trace_id:
                bucket["traces"].add(trace_id)
            usage = getattr(row, "usage_details", None)
            if not usage:
                bucket["missing"] += 1
            else:
                input_tokens = _tokens(usage, "input")
                output_tokens = _tokens(usage, "output")
                bucket["input"] += input_tokens
                bucket["output"] += output_tokens
                bucket["total"] += _tokens(usage, "total") or input_tokens + output_tokens
            cost = getattr(row, "total_cost", None)
            if cost is not None:
                bucket["cost"] += float(cost)
                bucket["has_cost"] = True
        cursor = getattr(page.meta, "cursor", None)
        if not cursor:
            break

    rows = [IntentUsage(
        intent=intent,
        requests=len(values["traces"]),
        input_tokens=values["input"],
        output_tokens=values["output"],
        total_tokens=values["total"],
        usage_missing=values["missing"],
        cost=values["cost"] if values["has_cost"] else None,
    ) for intent, values in buckets.items()]
    return sorted(rows, key=lambda row: (-row.total_tokens, row.intent))


class IntentUsageService:
    def __init__(self, client: Any):
        self.client = client

    def summarize(self, start: datetime, end: datetime) -> list[IntentUsage]:
        return summarize_usage(start, end, self.client)

