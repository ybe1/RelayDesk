"""Small chapter 5 end-to-end acceptance; never runs the chapter 4 dataset."""
import argparse
import asyncio
import json
from pathlib import Path
from uuid import uuid4
from app.db.session import SessionLocal
from app.schemas.chat import ChatRequest
from app.services.chat import DatabaseChatStore
from app.workflow.runtime import checkpoint_runtime
from app.workflow.graph import build_workflow
from app.workflow.service import WorkflowChatService

CASES = [("policy", "退货政策是什么"), ("logistics", "订单1001的物流到哪了"),
         ("complaint", "我要投诉"), ("smalltalk", "你好"),
         ("multi_step", "请先查订单1001的状态，再根据订单结果查物流，两步都需要")]


async def main(live):
    report_dir = Path(__file__).resolve().parents[1] / "data/ch05/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {"mode": "live" if live else "offline", "cases": []}
    async with checkpoint_runtime(report_dir / "acceptance.sqlite") as saver:
        graph = build_workflow(checkpointer=saver)
        for name, query in CASES:
            kwargs = {}
            if not live:
                from tests.test_ch05_agent_graph import GraphModel, tools_decision
                from tests.test_ch05_gate import result
                from langchain_core.messages import AIMessage
                intent = {"policy": "退款退货", "logistics": "物流", "complaint": "投诉", "multi_step": "物流"}.get(name)
                decisions = [AIMessage(content=json.dumps({"intent": intent}, ensure_ascii=False))] if intent else []
                if name in {"logistics", "multi_step"}:
                    if name == "multi_step": decisions.append(tools_decision("query_order", "a"))
                    decisions.append(tools_decision("query_logistics", "b"))
                decisions.append({"kind": "answer", "tool_calls": [], "actions": [], "reason": "足够"})
                model = GraphModel(decisions)
                async def retrieve(query, budget): return result()
                kwargs = {"model_factory": lambda: model, "retriever": retrieve}
            service = WorkflowChatService(graph, DatabaseChatStore(SessionLocal), **kwargs)
            req = ChatRequest(user_id="eval-ch05-" + uuid4().hex[:16], message=query)
            prepared = await service.prepare(req)
            events = []
            error = None
            try:
                async for event in service.events(req, prepared): events.append(event)
            except Exception as exc:
                error = type(exc).__name__ + ":" + str(exc)[:300]
            state = (await graph.aget_state({"configurable": {"thread_id": f"conversation:{prepared.conversation_id}"}})).values
            tool_calls = [call["name"] for msg in state.get("messages", []) for call in getattr(msg, "tool_calls", [])]
            trace = state.get("node_trace", [])
            checks = [error is None, any(kind == "done" for kind, _ in events)]
            if name in {"policy", "logistics", "multi_step"}:
                checks += [any(kind == "delta" and data.get("node") == "final_answer" for kind, data in events)]
            if name == "policy":
                kinds = [kind for kind, _ in events]
                checks += ["citations" in kinds and "delta" in kinds and kinds.index("citations") < kinds.index("delta")]
            if name == "policy": checks += ["retrieval" in trace, "confidence_gate" in trace, "final_answer" in trace, state.get("stop_reason") == "answered"]
            if name == "logistics": checks += ["query_logistics" in tool_calls, "retrieval" not in trace, state.get("stop_reason") == "answered"]
            if name == "complaint": checks += [len(state.get("actions", [])) == 2, "decision" not in trace]
            if name == "smalltalk": checks += [state.get("budget_usage", {}).get("model_calls") == 0]
            if name == "multi_step": checks += [tool_calls[:2] == ["query_order", "query_logistics"], "final_answer" in trace, state.get("stop_reason") == "answered"]
            row = {"case": name, "query": query, "conversation_id": prepared.conversation_id, "passed": all(checks),
                   "node_trace": trace, "tool_calls": tool_calls, "usage": state.get("budget_usage"),
                   "stop_reason": state.get("stop_reason"), "answer": state.get("final_text"), "error": error}
            report["cases"].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    report["passed"] = all(case["passed"] for case in report["cases"])
    (report_dir / ("acceptance-live.json" if live else "acceptance-offline.json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not report["passed"]: raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--live", action="store_true")
    modes.add_argument("--offline", action="store_true")
    asyncio.run(main(parser.parse_args().live))
