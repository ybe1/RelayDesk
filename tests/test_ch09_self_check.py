import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from app.services.tool_executor import ToolExecutor
from app.tools.business import build_tools
from app.workflow.budget import Limits, TurnBudget
from app.workflow.contracts import AgentDependencies, TurnContext
from tests.ch06_fakes import resolution_message
from tests.test_ch05_agent_graph import GraphModel
from tests.test_ch09_workflow_gate import config, retrieval


async def test_knowledge_self_check_blocks_answer_before_streaming():
    from app.workflow.graph import build_workflow, make_turn_input
    model = GraphModel([resolution_message("退货政策是什么"),
        AIMessage(content=json.dumps({"intent": "退款退货", "confidence": 0.95}, ensure_ascii=False))])
    tools = build_tools(lambda: None, 1)
    pooled = []
    async def retrieve(_q, _budget): return retrieval(0.95)
    async def check(_q, hits, _budget):
        assert hits and hits[0]["answer"] == "七日内"
        return SimpleNamespace(useful=False, reason="没有覆盖商品状态")
    async def pool(*args): pooled.append(args)
    context = TurnContext(AgentDependencies(model, tools, ToolExecutor(tools)),
        TurnBudget(Limits(tokens=100000)), retrieve=retrieve, evidence_check=check,
        record_low_confidence=pool, confidence_config=config(0.5))
    state = await build_workflow().ainvoke(make_turn_input("退货政策是什么", 1, "u", "t2",
        [HumanMessage(content="退货政策是什么")]), context=context)
    assert "evidence_self_check" in state["node_trace"]
    assert state["evidence_useful"] is False
    assert pooled[0][1] == "self_check"
    assert state["stop_reason"] == "fallback"
    assert model.calls == 2 and model.streams == 0
