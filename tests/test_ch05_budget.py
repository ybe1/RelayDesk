import pytest


def test_six_calls_allowed_seventh_rejected():
    from app.workflow.budget import BudgetExceeded, TurnBudget, Limits
    budget = TurnBudget(Limits(), clock=lambda: 0)
    for _ in range(6):
        budget.reserve_llm(10, 20)
        budget.settle_llm({"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    with pytest.raises(BudgetExceeded, match="model_calls"):
        budget.reserve_llm(10, 20)
    assert budget.snapshot()["model_calls"] == 6


def test_token_reservation_prevents_over_budget_request():
    from app.workflow.budget import BudgetExceeded, TurnBudget, Limits
    budget = TurnBudget(Limits(tokens=100), clock=lambda: 0)
    with pytest.raises(BudgetExceeded, match="tokens"):
        budget.reserve_llm(90, 11)
    assert budget.snapshot()["model_calls"] == 0


def test_missing_usage_charges_entire_reservation():
    from app.workflow.budget import TurnBudget, Limits
    budget = TurnBudget(Limits(), clock=lambda: 0)
    budget.reserve_llm(30, 40)
    budget.settle_llm(None)
    assert budget.snapshot()["tokens"] == 70
    assert budget.snapshot()["estimated"] is True


def test_deadline_and_tool_count_are_enforced():
    from app.workflow.budget import BudgetExceeded, TurnBudget, Limits
    now = [0]
    budget = TurnBudget(Limits(tools=2, seconds=10), clock=lambda: now[0])
    budget.reserve_tool()
    budget.reserve_tool()
    with pytest.raises(BudgetExceeded, match="tool_steps"):
        budget.reserve_tool()
    now[0] = 10
    with pytest.raises(BudgetExceeded, match="deadline"):
        budget.reserve_llm(1, 1)


def test_snapshot_restore_does_not_reset_usage_or_elapsed_time():
    from app.workflow.budget import TurnBudget, Limits
    now = [0]
    budget = TurnBudget(Limits(), clock=lambda: now[0])
    budget.reserve_llm(10, 20)
    budget.settle_llm(None)
    budget.reserve_tool()
    now[0] = 12
    restored = TurnBudget(Limits(), clock=lambda: 100, usage=budget.snapshot())
    assert restored.snapshot()["tokens"] == 30
    assert restored.snapshot()["tool_steps"] == 1
    assert restored.remaining_seconds() == 108


async def test_budgeted_model_limits_output_and_counts_failed_requests():
    from app.workflow.budget import TurnBudget, Limits
    from app.workflow.model import BudgetedModel
    from langchain_core.messages import HumanMessage
    class Broken:
        async def ainvoke(self, messages, **kwargs):
            assert kwargs["max_tokens"] <= 10
            raise ConnectionError("failure")
    budget = TurnBudget(Limits(), clock=lambda: 0)
    with pytest.raises(ConnectionError):
        await BudgetedModel(Broken(), budget, output_cap=10).ainvoke([HumanMessage(content="你好")])
    assert budget.snapshot()["model_calls"] == 1
    assert budget.snapshot()["estimated"]


async def test_stream_usage_is_settled_once():
    from app.workflow.budget import TurnBudget
    from app.workflow.model import BudgetedModel
    from langchain_core.messages import HumanMessage, AIMessageChunk
    class Streaming:
        async def astream(self, messages, **kwargs):
            yield AIMessageChunk(content="答")
            yield AIMessageChunk(content="", usage_metadata={
                "input_tokens": 3, "output_tokens": 1, "total_tokens": 4})
    budget = TurnBudget(clock=lambda: 0)
    chunks = [chunk async for chunk in BudgetedModel(Streaming(), budget).astream([HumanMessage(content="问题")])]
    assert chunks[0].content == "答"
    assert budget.snapshot()["tokens"] == 4
    assert budget.snapshot()["model_calls"] == 1


async def test_model_timeout_charges_request_and_stops():
    import asyncio
    from app.workflow.budget import TurnBudget, Limits, BudgetExceeded
    from app.workflow.model import BudgetedModel
    from langchain_core.messages import HumanMessage
    class Slow:
        async def ainvoke(self, messages, **kwargs):
            await asyncio.sleep(1)
    # Isolate request timeout from Windows scheduling delays before invocation.
    budget = TurnBudget(Limits(seconds=0.01), clock=lambda: 0)
    with pytest.raises(BudgetExceeded, match="deadline"):
        await BudgetedModel(Slow(), budget).ainvoke([HumanMessage(content="问题")])
    assert budget.snapshot()["model_calls"] == 1
    assert budget.snapshot()["tokens"] > 0


async def test_large_old_tool_turn_is_trimmed_before_budget_preflight():
    from app.workflow.budget import TurnBudget, Limits
    from app.workflow.model import BudgetedModel
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
    class Model:
        async def ainvoke(self, messages, **kwargs):
            assert [m.type for m in messages] == ['system', 'human']
            assert messages[-1].content == 'policy'
            return AIMessage(content='answer', usage_metadata={'input_tokens':10,'output_tokens':2,'total_tokens':12})
    messages = [SystemMessage(content='rules'), HumanMessage(content='old'),
        AIMessage(content='', tool_calls=[{'name':'query_faq','args':{},'id':'old'}]),
        ToolMessage(content='x'*8000,tool_call_id='old'), AIMessage(content='old answer'),
        HumanMessage(content='policy')]
    budget = TurnBudget(Limits(tokens=4000), clock=lambda:0)
    result = await BudgetedModel(Model(),budget,output_cap=100).ainvoke(messages)
    assert result.content == 'answer'
    assert len(messages) == 6
