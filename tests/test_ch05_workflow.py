import pytest
from langchain_core.messages import AIMessage, HumanMessage
from app.workflow.contracts import TurnContext, AgentDependencies
from app.workflow.budget import TurnBudget, Limits
from app.tools.business import build_tools
from app.services.tool_executor import ToolExecutor
from tests.test_ch05_agent_graph import GraphModel
from tests.test_ch05_gate import result
from tests.ch06_fakes import resolution_message


async def run_case(query, intent=None, score=0.8, pool_fails=False):
    from app.workflow.graph import build_workflow, make_turn_input
    responses = ([resolution_message(query),AIMessage(content='{"confidence":0.95,"intent":"'+intent+'"}')] if intent else [])
    responses += [{"kind": "answer", "tool_calls": [], "actions": [], "reason": "证据够"}]
    model = GraphModel(responses)
    retrieved, pooled, offers = [], [], []
    async def retrieve(question, budget):
        retrieved.append(question)
        return result(score)
    async def pool(question, reason):
        pooled.append(question)
        if pool_fails: raise ConnectionError("database")
    async def actions(types):
        offers.extend(types)
        return [{"action": t, "offer_id": t} for t in types]
    tools = build_tools(lambda: None, 1)
    ctx = TurnContext(AgentDependencies(model, tools, ToolExecutor(tools)), TurnBudget(Limits(tokens=100000)),
                      retrieve=retrieve, record_low_confidence=pool, issue_actions=actions)
    graph = build_workflow(gate_threshold=0.5)
    state = await graph.ainvoke(make_turn_input(query, 1, "user", "turn", [HumanMessage(content=query)]), context=ctx)
    return state, model, retrieved, pooled, offers


async def test_policy_cannot_skip_gate():
    state, model, retrieved, pooled, _ = await run_case("退货政策是什么", "退款退货", score=0.1)
    assert state["node_trace"] == ["resolve", "intent", "retrieval", "confidence_gate", "fallback", "log"]
    assert model.calls == 2 and model.streams == 0
    assert retrieved == pooled == ["退货政策是什么"]


async def test_business_has_no_pre_retrieval():
    state, model, retrieved, pooled, _ = await run_case("订单1001", "订单")
    assert not retrieved and not pooled
    assert "confidence_gate" not in state["node_trace"]
    assert model.streams == 1


async def test_complaint_offers_independent_actions_and_smalltalk_zero_calls():
    state, model, retrieved, _, offers = await run_case("我要投诉", "投诉")
    assert offers == ["handoff", "ticket"]
    assert model.calls == 2 and model.streams == 0 and not retrieved
    state, model, retrieved, pooled, offers = await run_case("你好")
    assert state["intent"] == "闲聊"
    assert model.calls == model.streams == 0
    assert not offers and not retrieved and not pooled


async def test_pool_failure_does_not_claim_persisted(caplog):
    state, *_ = await run_case("邮费？", "退款退货", score=0.1, pool_fails=True)
    assert "low-confidence persistence failed" in caplog.text
    assert "已记录" not in state["final_text"]


def test_turn_input_resets_evidence_and_actions():
    from app.workflow.graph import make_turn_input
    state = make_turn_input("订单1001", 1, "user", "new", [])
    assert state["citations"] == state["evidence"] == state["actions"] == []
    assert state["budget_usage"] == {}
