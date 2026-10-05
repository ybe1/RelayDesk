import json
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from tests.ch05_fakes import bare_scenario, SequenceModel


async def test_bare_loop_uses_second_observation():
    result, calls = await bare_scenario(["query_order", "query_logistics"])
    assert calls == ["query_order", "query_logistics"]
    assert result.stop_reason == "answered"
    assert result.messages[-1].content == "演示物流结果"


async def test_tool_limit_closes_all_pending_calls_without_another_model_call():
    from app.workflow.bare_loop import run_bare_loop
    from app.workflow.contracts import AgentDependencies
    from app.workflow.budget import TurnBudget, Limits
    from app.tools.business import build_tools
    from tests.test_tool_executor import ToolExecutor
    model = SequenceModel([AIMessage(content="", tool_calls=[
        {"name": "query_order", "args": {"order_id": "1001"}, "id": "a"},
        {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "b"}])])
    tools = build_tools(lambda: None, 1)
    result = await run_bare_loop([HumanMessage(content="查订单")], AgentDependencies(model, tools, ToolExecutor(tools)),
                                 TurnBudget(Limits(tools=1), clock=lambda: 0))
    assert result.stop_reason == "tool_steps"
    assert len(model.requests) == 1
    results = [m for m in result.messages if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in results] == ["a", "b"]
    assert json.loads(results[1].content)["error"]["code"] == "tool_steps"


async def test_agent_cannot_execute_create_ticket_even_if_model_requests_it():
    from app.workflow.bare_loop import run_bare_loop
    from app.workflow.contracts import AgentDependencies
    from app.workflow.budget import TurnBudget, Limits
    from app.tools.business import build_tools
    from tests.test_tool_executor import ToolExecutor
    model = SequenceModel([AIMessage(content="", tool_calls=[
        {"name": "create_ticket", "args": {"description": "投诉", "ticket_type": "投诉"}, "id": "x"}]),
        AIMessage(content="请点击建工单")])
    tools = build_tools(lambda: (_ for _ in ()).throw(AssertionError("must not write")), 1)
    result = await run_bare_loop([HumanMessage(content="投诉")], AgentDependencies(model, tools, ToolExecutor(tools)),
                                 TurnBudget(Limits(), clock=lambda: 0))
    tool_result = next(m for m in result.messages if isinstance(m, ToolMessage))
    assert json.loads(tool_result.content)["error"]["code"] == "permission_denied"
