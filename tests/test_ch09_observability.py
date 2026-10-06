import asyncio
from contextlib import contextmanager

import pytest

from app.observability.langfuse import ObservabilityRuntime
from app.observability.spans import observe_external, use_runtime


class Handler:
    raise_error = True
    last_trace_id = "trace-1"


class Observation:
    def __init__(self, owner, kwargs): self.owner, self.kwargs = owner, kwargs
    def update(self, **kwargs): self.owner.updates.append(kwargs)


class Client:
    def __init__(self): self.started=[]; self.updates=[]; self.tags=[]; self.flushed=0; self.shutdowns=0
    @contextmanager
    def start_as_current_observation(self, **kwargs):
        self.started.append(kwargs)
        yield Observation(self, kwargs)
    def get_current_trace_id(self): return "trace-1"
    def _create_trace_tags_via_ingestion(self, *, trace_id, tags): self.tags.append((trace_id, tags))
    def flush(self): self.flushed += 1
    def shutdown(self): self.shutdowns += 1


def test_trace_config_is_request_scoped_and_callback_bound_once():
    runtime = ObservabilityRuntime(enabled=True, client=Client(), handler=Handler())
    first = runtime.trace_config(7, "turn-a")
    second = runtime.trace_config(8, "turn-b")
    assert first is not second
    assert first["callbacks"].count(runtime.handler) == 1
    assert first["metadata"]["langfuse_session_id"] == "conversation:7"
    assert first["metadata"]["turn_id"] == "turn-a"
    assert runtime.handler.raise_error is False


def test_disabled_runtime_has_no_callback_or_export_side_effects():
    client = Client(); runtime = ObservabilityRuntime(enabled=False, client=client, handler=Handler())
    assert runtime.trace_config(1, "t") == {}
    runtime.tag_intent("退款退货")
    runtime.flush(0.01)
    assert not client.tags and client.flushed == 0


@pytest.mark.asyncio
async def test_external_observation_records_result_error_and_never_breaks_business_call():
    client = Client(); runtime = ObservabilityRuntime(enabled=True, client=client, handler=Handler())
    with use_runtime(runtime):
        async with observe_external("milvus.hybrid", {"query": "邮费"}, {"attempt": 1}) as span:
            span.finish(output={"hits": 2}, metadata={"duration_ms": 12})
    assert client.started[0]["name"] == "milvus.hybrid"
    assert client.updates[-1]["output"] == {"hits": 2}

    class BrokenClient(Client):
        def start_as_current_observation(self, **kwargs): raise RuntimeError("collector down")
    with use_runtime(ObservabilityRuntime(enabled=True, client=BrokenClient(), handler=Handler())):
        async with observe_external("tool.query", {}, {}):
            value = 42
    assert value == 42


def test_intent_tag_and_bounded_flush_are_failure_isolated():
    client = Client(); runtime = ObservabilityRuntime(enabled=True, client=client, handler=Handler())
    runtime.tag_intent("物流")
    assert client.tags == [("trace-1", ["intent:物流"])]
    runtime.flush(0.2)
    assert client.flushed == 1

    class Slow(Client):
        def flush(self):
            import time; time.sleep(0.2)
    import time
    started = time.monotonic()
    ObservabilityRuntime(enabled=True, client=Slow(), handler=Handler()).flush(0.01)
    assert time.monotonic() - started < 0.15
