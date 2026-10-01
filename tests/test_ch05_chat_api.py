import pytest
from app.schemas.chat import ChatRequest
from app.services.chat import DatabaseChatStore
from app.db.session import SessionLocal
from tests.test_ch05_agent_graph import GraphModel


async def test_prepare_rejects_concurrent_request_and_wrong_owner():
    from app.workflow.service import WorkflowChatService
    from app.workflow.graph import build_workflow
    service = WorkflowChatService(build_workflow(), DatabaseChatStore(SessionLocal), model_factory=lambda: GraphModel([]))
    req = ChatRequest(user_id="ch05-prepare", message="你好")
    prepared = await service.prepare(req)
    cid = prepared.conversation_id
    try:
        with pytest.raises(RuntimeError, match="busy"):
            await service.prepare(ChatRequest(user_id=req.user_id, conversation_id=cid, message="你好"))
        with pytest.raises(PermissionError):
            await service.prepare(ChatRequest(user_id="other", conversation_id=cid, message="你好"))
    finally:
        service.release(prepared)


async def test_greeting_service_uses_existing_sse_contract():
    from app.workflow.service import WorkflowChatService
    from app.workflow.graph import build_workflow
    model = GraphModel([])
    service = WorkflowChatService(build_workflow(), DatabaseChatStore(SessionLocal), model_factory=lambda: model)
    events = [event async for event in service.events(ChatRequest(user_id="ch05-greeting", message="你好"))]
    assert events[0][0] == "conversation"
    assert events[-1][0] == "done"
    assert any(kind == "delta" for kind, _ in events)
    assert model.calls == 0


async def test_cancel_closes_persisted_tool_and_releases_lock(caplog):
    import asyncio
    from app.workflow.service import WorkflowChatService
    from app.workflow.graph import build_workflow
    from langchain_core.messages import AIMessage
    from app.repositories.messages import load_messages
    from sqlalchemy import select
    from app.db.models import Conversation
    entered = asyncio.Event()
    class BlockingGraph:
        async def astream(self, state, config, context, **kwargs):
            await context.save_message(AIMessage(content="", id=state["turn_id"]+":decision:1", tool_calls=[
                {"name": "query_order", "args": {"order_id": "1001"}, "id": "cancel-call"}]))
            entered.set()
            await asyncio.sleep(100)
            yield ("done", None)
    service = WorkflowChatService(BlockingGraph(), DatabaseChatStore(SessionLocal), model_factory=lambda: GraphModel([]))
    req = ChatRequest(user_id="ch05-cancel", message="物流")
    prepared = await service.prepare(req)
    async def consume():
        async for _ in service.events(req, prepared): pass
    task = asyncio.create_task(consume())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    assert prepared.conversation_id not in service.active
    assert "workflow interrupted" in caplog.text
    with SessionLocal() as session:
        rows = load_messages(session, prepared.conversation_id)
        assert rows[-1].role == "tool"
        assert rows[-1].tool_call_id == "cancel-call"


def test_lifespan_opens_and_closes_checkpointer_and_api_statuses():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        service = app.state.workflow_service
        assert service is not None
        response = client.post("/api/chat", json={"user_id": "ch05-http", "message": "你好"})
        assert response.status_code == 200
        import re
        cid = int(re.search(r'"conversation_id": (\d+)', response.text).group(1))
        assert client.post("/api/chat", json={"user_id": "other-http", "conversation_id": cid, "message": "你好"}).status_code == 403
        service.active.add(cid)
        assert client.post("/api/chat", json={"user_id": "ch05-http", "conversation_id": cid, "message": "你好"}).status_code == 409
        service.active.discard(cid)
    assert app.state.workflow_service is None


async def test_business_service_streams_child_graph_deltas():
    from app.workflow.service import WorkflowChatService
    from app.workflow.graph import build_workflow
    from langchain_core.messages import AIMessage
    from tests.test_ch05_agent_graph import tools_decision
    model = GraphModel([AIMessage(content='{"intent":"物流"}'), tools_decision("query_logistics"),
                        {"kind": "answer", "tool_calls": [], "actions": [], "reason": "足够"}])
    service = WorkflowChatService(build_workflow(), DatabaseChatStore(SessionLocal), model_factory=lambda: model)
    events = [event async for event in service.events(ChatRequest(user_id="ch05-child-stream", message="订单1001物流"))]
    assert [data["delta"] for kind, data in events if kind == "delta"] == ["演示", "物流结果"]


async def test_policy_service_citations_precede_child_answer():
    from app.workflow.service import WorkflowChatService
    from app.workflow.graph import build_workflow
    from langchain_core.messages import AIMessage
    from tests.test_ch05_gate import result
    model = GraphModel([AIMessage(content='{"intent":"退款退货"}'),
                        {"kind": "answer", "tool_calls": [], "actions": [], "reason": "足够"}])
    async def retrieve(query, budget): return result()
    service = WorkflowChatService(build_workflow(), DatabaseChatStore(SessionLocal), model_factory=lambda: model, retriever=retrieve)
    events = [event async for event in service.events(ChatRequest(user_id="ch05-child-policy", message="邮费是多少"))]
    kinds = [kind for kind, _ in events]
    assert kinds.index("citations") < kinds.index("delta")


async def test_policy_then_business_round_resets_checkpoint_evidence(tmp_path):
    from app.workflow.runtime import checkpoint_runtime
    from app.workflow.service import WorkflowChatService
    from app.workflow.graph import build_workflow
    from langchain_core.messages import AIMessage
    from tests.test_ch05_gate import result
    from tests.test_ch05_agent_graph import tools_decision
    models = iter([
        GraphModel([AIMessage(content='{"intent":"退款退货"}'), {"kind": "answer", "tool_calls": [], "actions": [], "reason": "足够"}]),
        GraphModel([AIMessage(content='{"intent":"物流"}'), tools_decision("query_logistics"), {"kind": "answer", "tool_calls": [], "actions": [], "reason": "足够"}])])
    async def retrieve(query, budget): return result()
    async with checkpoint_runtime(tmp_path / "rounds.sqlite") as saver:
        graph = build_workflow(checkpointer=saver)
        service = WorkflowChatService(graph, DatabaseChatStore(SessionLocal), model_factory=lambda: next(models), retriever=retrieve)
        events = [e async for e in service.events(ChatRequest(user_id="ch05-rounds", message="退货政策"))]
        cid = events[0][1]["conversation_id"]
        events = [e async for e in service.events(ChatRequest(user_id="ch05-rounds", conversation_id=cid, message="订单1001物流"))]
        assert not any(kind == "citations" for kind, _ in events)
        state = (await graph.aget_state({"configurable": {"thread_id": f"conversation:{cid}"}})).values
        assert state["evidence"] == state["citations"] == state["actions"] == []
        assert "retrieval" not in state["node_trace"]
        assert state["budget_usage"]["model_calls"] == 4
