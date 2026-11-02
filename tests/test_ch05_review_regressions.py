import asyncio
import threading
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from app.services.tool_executor import ToolExecutor
from app.workflow.budget import TurnBudget, Limits
from app.workflow.bare_loop import run_bare_loop
from app.workflow.contracts import AgentDependencies
from tests.ch05_fakes import SequenceModel


async def test_retry_attempts_consume_shared_tool_budget():
    calls = []
    @tool
    async def query_order(order_id: str) -> str:
        """Read an order."""
        calls.append(order_id)
        if len(calls) % 3:
            raise ConnectionError("retry")
        return "ok"
    requests = [{"name": "query_order", "args": {"order_id": str(i)}, "id": str(i)} for i in range(5)]
    model = SequenceModel([AIMessage(content="", tool_calls=requests), AIMessage(content="done")])
    budget = TurnBudget(Limits(tools=5), clock=lambda: 0)
    result = await run_bare_loop([HumanMessage(content="orders")],
        AgentDependencies(model, [query_order], ToolExecutor([query_order], backoff_seconds=0)), budget)
    assert len(calls) == budget.tool_steps == 5
    assert result.stop_reason == "tool_steps"


async def test_cancel_waits_for_inflight_save_before_cleanup_and_release():
    from app.workflow.service import WorkflowChatService, PreparedTurn
    from app.schemas.chat import ChatRequest
    entered, unblock = threading.Event(), threading.Event()
    rows = []
    class Store:
        session_factory = None
        def load_history(self, cid): return []
    class Graph:
        async def astream(self, state, config, context, **kwargs):
            await context.save_message(AIMessage(content="", tool_calls=[
                {"name": "query_order", "args": {"order_id": "1"}, "id": "late"}]))
            yield (), ("done", None)
    service = WorkflowChatService(Graph(), Store(), model_factory=lambda: None, tools_factory=lambda *a, **k: [])
    def save(cid, turn, msg):
        if msg.type == "ai":
            entered.set()
            assert unblock.wait(5)
        rows.append(msg.type)
    service._save = save
    prepared = PreparedTurn(1, "turn")
    service.active.add(1)
    async def consume():
        async for _ in service.events(ChatRequest(user_id="u", message="order"), prepared): pass
    task = asyncio.create_task(consume())
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.sleep(0.05)
    try:
        assert 1 in service.active
        assert not task.done()
    finally:
        unblock.set()
        with pytest.raises(asyncio.CancelledError): await task
    assert rows == ["human", "ai"]
    assert 1 not in service.active


async def test_graph_retry_attempts_consume_budget_and_close_all_requests():
    from app.workflow.agent import build_agent_graph
    from app.workflow.contracts import TurnContext
    from tests.test_ch05_agent_graph import GraphModel
    calls = []
    @tool
    async def query_order(order_id: str) -> str:
        """Read an order."""
        calls.append(order_id)
        raise ConnectionError("retry")
    requests = [{"name": "query_order", "args": {"order_id": str(i)}, "id": str(i)} for i in range(3)]
    model = GraphModel([AIMessage(content="", tool_calls=requests)])
    budget = TurnBudget(Limits(tools=2), clock=lambda: 0)
    saved = []
    async def save(msg): saved.append(msg)
    context = TurnContext(AgentDependencies(model, [query_order], ToolExecutor([query_order], backoff_seconds=0)), budget, save)
    state = await build_agent_graph().ainvoke({"messages": [HumanMessage(content="orders")], "turn_id": "retry",
        "node_trace": [], "citations": [], "evidence": [], "actions": []}, context=context)
    assert len(calls) == budget.tool_steps == 2
    assert state["stop_reason"] == "tool_steps"
    assert len([m for m in saved if m.type == "tool"]) == 3
