import json

from langchain_core.messages import AIMessage, HumanMessage

from app.flywheel.contracts import ConfidenceConfig
from app.knowledge.query_understanding import QueryUnderstanding
from app.knowledge.retrieval import RetrievalResult
from app.services.tool_executor import ToolExecutor
from app.tools.business import build_tools
from app.workflow.budget import Limits, TurnBudget
from app.workflow.contracts import AgentDependencies, TurnContext
from tests.ch06_fakes import resolution_message
from tests.test_ch05_agent_graph import GraphModel


def config(threshold=0.7):
    return ConfidenceConfig(threshold=threshold, top1_floor=0, top1_ceiling=1,
        gap_floor=0, gap_ceiling=1, effective_score_threshold=0.5,
        top_k=5, calibration_id="test")


def retrieval(score=0.4):
    return RetrievalResult(QueryUnderstanding(normalized_query="退货", search_query="退货"), [{
        "id": 7, "rerank_score": score, "question": "退货政策", "answer": "七日内",
        "section_path": "退货/时限", "is_key_clause": True,
    }], True, "ok")


async def _run(retrieve, pool):
    from app.workflow.graph import build_workflow, make_turn_input
    model = GraphModel([resolution_message("退货政策是什么"),
        AIMessage(content=json.dumps({"intent": "退款退货", "confidence": 0.95}, ensure_ascii=False))])
    tools = build_tools(lambda: None, 1)
    context = TurnContext(AgentDependencies(model, tools, ToolExecutor(tools)),
        TurnBudget(Limits(tokens=100000)), retrieve=retrieve,
        record_low_confidence=pool, confidence_config=config())
    return await build_workflow().ainvoke(make_turn_input("退货政策是什么", 1, "u", "t",
        [HumanMessage(content="退货政策是什么")]), context=context)


async def test_rejected_gate_keeps_snapshot_but_clears_answer_evidence():
    calls = []
    async def retrieve(_q, _budget): return retrieval()
    async def pool(*args): calls.append(args)
    state = await _run(retrieve, pool)
    assert state["gate_allowed"] is False
    assert state["evidence"] == []
    assert state["evidence_snapshot"]["hits"][0]["answer"] == "七日内"
    assert state["confidence"]["reason"] == "evidence_below_threshold"
    assert calls[0][1] == "retrieval_low_conf"
    final = next(m for m in state["messages"] if m.id == "t:final")
    assert final.additional_kwargs["evidence_snapshot"]["retrieval_id"]


async def test_retrieval_failure_is_operational_and_never_enters_gap_pool():
    calls = []
    async def retrieve(_q, _budget): raise TimeoutError("network")
    async def pool(*args): calls.append(args)
    state = await _run(retrieve, pool)
    assert state["gate_reason"] == "retrieval_unavailable:TimeoutError"
    assert calls == []
    assert state["evidence_snapshot"] == {}
