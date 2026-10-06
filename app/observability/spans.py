from __future__ import annotations

import logging
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar


LOGGER = logging.getLogger(__name__)
_RUNTIME: ContextVar[object | None] = ContextVar("mewhelp_observability", default=None)


@contextmanager
def use_runtime(runtime):
    token = _RUNTIME.set(runtime)
    try:
        yield
    finally:
        _RUNTIME.reset(token)


class ExternalSpan:
    def __init__(self, observation=None): self.observation = observation

    def finish(self, *, output=None, metadata=None):
        if self.observation is None:
            return
        try:
            self.observation.update(output=output, metadata=metadata)
        except Exception:
            LOGGER.exception("Langfuse external observation update failed")


@asynccontextmanager
async def observe_external(name, input, metadata=None, *, as_type="span"):
    runtime = _RUNTIME.get()
    if runtime is None or not getattr(runtime, "enabled", False) or runtime.client is None:
        yield ExternalSpan()
        return
    manager = None
    try:
        manager = runtime.client.start_as_current_observation(
            name=name, as_type=as_type, input=input, metadata=metadata or {})
        observation = manager.__enter__()
    except Exception:
        LOGGER.exception("Langfuse external observation start failed name=%s", name)
        yield ExternalSpan()
        return
    try:
        yield ExternalSpan(observation)
    except BaseException as exc:
        try:
            observation.update(level="ERROR", status_message=type(exc).__name__)
        except Exception:
            LOGGER.exception("Langfuse external observation error update failed")
        raise
    finally:
        try:
            manager.__exit__(None, None, None)
        except Exception:
            LOGGER.exception("Langfuse external observation close failed")
