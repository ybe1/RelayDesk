import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.chat import get_chat_service, router
from app.workflow.service import PreparedTurn


def client_for(service):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_chat_service] = lambda: service
    return TestClient(app)


class PreparedService:
    def __init__(self):
        self.prepared = PreparedTurn(7, "turn")
        self.released = []

    async def prepare(self, req):
        return self.prepared

    async def events(self, req, prepared=None):
        assert prepared is self.prepared
        yield "conversation", {"conversation_id": 7}
        yield "delta", {"delta": "你好"}
        yield "done", None

    def release(self, prepared):
        self.released.append(prepared)


def test_chat_always_prepares_workflow_turn_and_releases_it():
    service = PreparedService()
    response = client_for(service).post("/api/chat", json={"user_id": "u", "message": "你好"})
    assert response.status_code == 200
    assert "event: conversation" in response.text
    assert "你好" in response.text and "data: [DONE]" in response.text
    assert service.released == [service.prepared]


@pytest.mark.parametrize("error,status", [(PermissionError, 403), (LookupError, 404), (RuntimeError, 409)])
def test_prepare_errors_are_http_errors_before_streaming(error, status):
    class Service(PreparedService):
        async def prepare(self, req):
            raise error("private detail")
    service = Service()
    response = client_for(service).post("/api/chat", json={"user_id": "u", "message": "你好"})
    assert response.status_code == status
    assert "private detail" not in response.text
    assert not service.released


def test_stream_failure_releases_prepared_turn_without_leaking_details():
    class Service(PreparedService):
        async def events(self, req, prepared=None):
            assert prepared is self.prepared
            raise ValueError("private detail")
            yield
    service = Service()
    response = client_for(service).post("/api/chat", json={"user_id": "u", "message": "你好"})
    assert "event: error" in response.text
    assert "private detail" not in response.text
    assert service.released == [service.prepared]


def test_current_workflow_greeting_needs_no_model_and_releases_session():
    from tests.test_ch05_agent_graph import GraphModel
    from tests.workflow_fixture import OfflineWorkflowService
    model = GraphModel([])
    service = OfflineWorkflowService(model)
    response = client_for(service).post("/api/chat", json={"user_id": "u", "message": "你好"})
    assert "data: [DONE]" in response.text
    assert "我是小喵" in response.text
    assert model.calls == model.streams == 0
    assert [m.type for m in service.store.rows] == ["human", "ai"]
    assert not service.active


def test_current_workflow_streams_and_persists_citations_before_answer():
    from tests.ch06_fakes import resolution_message
    from tests.test_ch05_agent_graph import GraphModel
    from tests.test_ch05_gate import result
    from tests.workflow_fixture import OfflineWorkflowService
    from langchain_core.messages import AIMessage
    model = GraphModel([resolution_message("邮费是多少"),
                        AIMessage(content='{"intent":"商品咨询","confidence":0.99}'),
                        {"kind": "answer", "reason": "证据足够"}])
    async def retrieve(query, budget):
        return result()
    service = OfflineWorkflowService(model, retriever=retrieve)
    response = client_for(service).post("/api/chat", json={"user_id": "u", "message": "邮费是多少"})
    assert response.status_code == 200
    assert response.text.index("event: citations") < response.text.index('data: {"delta":')
    assert service.store.rows[-1].additional_kwargs["citations"][0]["chunk_id"] == 1
    assert "data: [DONE]" in response.text and not service.active
