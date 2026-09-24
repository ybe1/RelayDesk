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


def test_real_chat_service_sse_citations_precede_answer_and_survive_persistence():
    import json
    from langchain_core.messages import AIMessage, ToolMessage
    from app.services.chat import ChatService
    from app.services.tool_executor import ToolExecutor
    from app.tools.business import build_tools
    from tests.test_chat_service import Model, MemoryStore
    store = MemoryStore()
    model = Model(AIMessage(content="", tool_calls=[{"name": "query_faq", "args": {"keyword": "邮费"}, "id": "citation"}]))
    service = ChatService(model, store, lambda cid: build_tools(lambda: None, cid, lambda _: [{"id": 9, "question": "邮费规则", "answer": "满99元包邮", "category": "配送"}]), ToolExecutor)
    app.dependency_overrides[chat_api.get_chat_service] = lambda: service
    response = TestClient(app).post("/api/chat", json={"user_id": "u", "message": "邮费"})
    body = response.text
    assert body.index("event: citations") < body.index('data: {"delta":')
    persisted = next(row for row in store.rows if isinstance(row, ToolMessage))
    citation = json.loads(json.loads(persisted.content)["result"])["citations"][0]
    assert citation["chunk_id"] == 9
    assert json.dumps(citation, ensure_ascii=False) in body
    assert "data: [DONE]" in body
