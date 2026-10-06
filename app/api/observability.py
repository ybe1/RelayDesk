from __future__ import annotations

import logging
import threading
import time
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, HTTPException, Query, Request

from app.observability.usage import IntentUsageService


LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/api/observability", tags=["observability"])
_CACHE: dict[tuple[int, int], tuple[float, dict]] = {}
_CACHE_LOCK = threading.Lock()


def _service(request: Request):
    injected = getattr(request.app.state, "intent_usage_service", None)
    if injected is not None:
        return injected
    runtime = getattr(request.app.state, "observability", None)
    if runtime is None or not runtime.enabled or runtime.client is None:
        raise HTTPException(503, "Langfuse usage unavailable")
    return IntentUsageService(runtime.client)


@router.get("/intent-usage")
def intent_usage(request: Request, days: int = Query(7, ge=1, le=90)):
    service = _service(request)
    key = (id(service), days)
    now_monotonic = time.monotonic()
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached and now_monotonic - cached[0] < 30:
            return cached[1]
    end = datetime.now(UTC)
    start = end - timedelta(days=days)
    try:
        rows = service.summarize(start, end)
    except Exception as error:
        LOGGER.warning("Langfuse usage unavailable: %s", type(error).__name__)
        raise HTTPException(503, "Langfuse usage unavailable") from error
    result = {
        "status": "ok",
        "days": days,
        "from": start.isoformat(),
        "to": end.isoformat(),
        "items": [row.model_dump(mode="json") for row in rows],
    }
    with _CACHE_LOCK:
        _CACHE[key] = (now_monotonic, result)
    return result

