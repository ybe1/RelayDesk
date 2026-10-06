from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.observability import router


class UsageService:
    def __init__(self, *, result=None, error=None):
        self.result = result or []
        self.error = error
        self.calls = 0

    def summarize(self, start, end):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


def make_client(service):
    app = FastAPI()
    app.include_router(router)
    app.state.intent_usage_service = service
    return TestClient(app)


def test_intent_usage_has_bounded_days_and_30_second_cache():
    service = UsageService(result=[SimpleNamespace(model_dump=lambda mode=None: {
        "intent": "物流", "requests": 2, "input_tokens": 10, "output_tokens": 3,
        "total_tokens": 13, "usage_missing": 0, "cost": 0.1,
    })])
    client = make_client(service)

    first = client.get("/api/observability/intent-usage?days=7")
    second = client.get("/api/observability/intent-usage?days=7")

    assert first.status_code == 200
    assert first.json()["status"] == "ok"
    assert first.json()["items"][0]["intent"] == "物流"
    assert second.json() == first.json()
    assert service.calls == 1
    assert client.get("/api/observability/intent-usage?days=0").status_code == 422
    assert client.get("/api/observability/intent-usage?days=91").status_code == 422


def test_intent_usage_reports_upstream_unavailable_without_fake_zero():
    response = make_client(UsageService(error=RuntimeError("collector down"))).get(
        "/api/observability/intent-usage?days=7"
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Langfuse usage unavailable"

