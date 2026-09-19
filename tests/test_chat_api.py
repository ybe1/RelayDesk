from fastapi.testclient import TestClient

from app.api import chat as chat_api
from app.main import app


class FakeService:
    async def events(self, req):
        yield "conversation", {"conversation_id": 7}
        yield "tool_status", {"tool_name": "query_faq", "phase": "running", "message": "正在查询"}
        yield "delta", {"delta": "你"}
        yield "delta", {"delta": "好"}
        yield "done", None


def teardown_function():
    app.dependency_overrides.clear()


def test_chat_streams_conversation_tool_status_tokens_and_done():
    app.dependency_overrides[chat_api.get_chat_service] = lambda: FakeService()
    with TestClient(app).stream("POST", "/api/chat", json={"user_id": "anon-1", "message": "退货政策"}) as resp:
        assert resp.status_code == 200
        body = "\n".join(resp.iter_lines())
    assert 'event: conversation\ndata: {"conversation_id": 7}' in body
    assert 'event: tool_status\ndata: ' in body
    assert '"tool_name": "query_faq"' in body
    assert 'data: {"delta": "你"}' in body
    assert 'data: {"delta": "好"}' in body
    assert "data: [DONE]" in body


def test_chat_validates_empty_message():
    app.dependency_overrides[chat_api.get_chat_service] = lambda: FakeService()
    assert TestClient(app).post("/api/chat", json={"user_id": "anon-1", "message": ""}).status_code == 422
