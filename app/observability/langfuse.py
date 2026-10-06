from __future__ import annotations

import logging
import threading
from typing import Any


LOGGER = logging.getLogger(__name__)


class ObservabilityRuntime:
    """Failure-isolated Langfuse integration; disabled mode is a true no-op."""

    def __init__(self, *, enabled: bool, host: str | None = None, public_key: str | None = None,
                 secret_key: str | None = None, client: Any = None, handler: Any = None):
        self.enabled = bool(enabled)
        self.client = client
        self.handler = handler
        if self.enabled and (not host or not public_key or not secret_key) and client is None:
            LOGGER.warning("Langfuse disabled because host or keys are missing")
            self.enabled = False
        if self.enabled and self.client is None:
            try:
                from langfuse import Langfuse
                self.client = Langfuse(host=host, public_key=public_key, secret_key=secret_key,
                                       tracing_enabled=True, timeout=5)
            except Exception:
                LOGGER.exception("Langfuse client initialization failed")
                self.enabled = False
        if self.enabled and self.handler is None:
            try:
                from langfuse.langchain import CallbackHandler
                self.handler = CallbackHandler(public_key=public_key)
            except Exception:
                LOGGER.exception("Langfuse callback initialization failed")
                self.enabled = False
        if self.handler is not None:
            # LangChain logs callback failures and continues the business path.
            self.handler.raise_error = False

    @classmethod
    def from_settings(cls, settings):
        return cls(enabled=settings.langfuse_enabled, host=settings.langfuse_host,
                   public_key=settings.langfuse_public_key, secret_key=settings.langfuse_secret_key)

    def trace_config(self, conversation_id: int, turn_id: str) -> dict:
        if not self.enabled or self.handler is None:
            return {}
        return {"callbacks": [self.handler], "metadata": {
            "langfuse_session_id": f"conversation:{conversation_id}",
            "langfuse_trace_name": "customer-service-turn",
            "langfuse_tags": ["online"],
            "conversation_id": conversation_id,
            "turn_id": turn_id,
        }}

    def tag_intent(self, intent: str) -> None:
        if not self.enabled or self.client is None:
            return
        try:
            trace_id = self.client.get_current_trace_id()
            if trace_id:
                # SDK 4.17 has no public late trace-tag updater. This exact-version
                # ingestion helper is isolated here and guarded for future SDK changes.
                updater = getattr(self.client, "_create_trace_tags_via_ingestion", None)
                if updater is not None:
                    updater(trace_id=trace_id, tags=[f"intent:{intent}"])
            update_span = getattr(self.client, "update_current_span", None)
            if update_span is not None:
                update_span(metadata={"intent": intent})
        except Exception:
            LOGGER.exception("Langfuse intent tagging failed")

    def flush(self, timeout_seconds: float = 2.0) -> None:
        if not self.enabled or self.client is None:
            return
        done = threading.Event()

        def export():
            try:
                self.client.flush()
            except Exception:
                LOGGER.exception("Langfuse flush failed")
            finally:
                done.set()

        threading.Thread(target=export, daemon=True, name="langfuse-flush").start()
        if not done.wait(max(0.0, timeout_seconds)):
            LOGGER.warning("Langfuse flush exceeded %.2fs", timeout_seconds)

    def shutdown(self, timeout_seconds: float = 2.0) -> None:
        self.flush(timeout_seconds)
        if self.enabled and self.client is not None:
            try:
                self.client.shutdown()
            except Exception:
                LOGGER.exception("Langfuse shutdown failed")
