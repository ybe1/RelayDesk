from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from app.observability.usage import summarize_usage


def observation(*, trace_id, tags, type="GENERATION", usage=None, cost=None):
    return SimpleNamespace(
        trace_id=trace_id,
        tags=tags,
        type=type,
        usage_details=usage,
        total_cost=cost,
    )


class Observations:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def get_many(self, **kwargs):
        self.calls.append(kwargs)
        cursor = kwargs.get("cursor")
        return self.pages[cursor]


def client(pages):
    return SimpleNamespace(api=SimpleNamespace(observations=Observations(pages)))


def page(rows, cursor=None):
    return SimpleNamespace(data=rows, meta=SimpleNamespace(cursor=cursor))


def window():
    end = datetime(2026, 10, 6, tzinfo=UTC)
    return end - timedelta(days=7), end


def test_usage_pages_generations_and_does_not_double_count_parent_or_offline():
    start, end = window()
    fake = client({
        None: page([
            observation(trace_id="a", tags=["online", "intent:物流"], usage={"input": 600, "output": 100, "total": 700}, cost=0.02),
            observation(trace_id="a", tags=["online", "intent:物流"], type="CHAIN", usage={"input": 1000, "output": 0}),
            observation(trace_id="off", tags=["offline", "intent:物流"], usage={"input": 999, "output": 999}),
        ], "next"),
        "next": page([
            observation(trace_id="a", tags=["online", "intent:物流"], usage={"input": 40, "output": 10}, cost=0.01),
            observation(trace_id="b", tags=["online", "intent:订单"], usage=None),
        ]),
    })

    rows = {row.intent: row for row in summarize_usage(start, end, fake)}

    assert rows["物流"].requests == 1
    assert rows["物流"].input_tokens == 640
    assert rows["物流"].output_tokens == 110
    assert rows["物流"].total_tokens == 750
    assert rows["物流"].cost == 0.03
    assert rows["订单"].requests == 1
    assert rows["订单"].usage_missing == 1
    calls = fake.api.observations.calls
    assert [call.get("cursor") for call in calls] == [None, "next"]
    assert calls[0]["fields"] == "core,basic,usage,trace_context"
    assert calls[0]["type"] == "GENERATION"
    assert calls[0]["from_start_time"] == start
    assert calls[0]["to_start_time"] == end


def test_online_generation_without_intent_is_reported_as_unclassified():
    start, end = window()
    fake = client({None: page([
        observation(trace_id="x", tags=["online"], usage={"input_tokens": 2, "output_tokens": 3}),
    ])})

    rows = summarize_usage(start, end, fake)

    assert len(rows) == 1
    assert rows[0].intent == "未分类"
    assert rows[0].total_tokens == 5

