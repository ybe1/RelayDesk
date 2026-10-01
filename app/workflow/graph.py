"""Deterministic parent workflow; model labels cannot override routing."""
import asyncio
import logging
from langchain_core.messages import AIMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime
from .contracts import WorkflowState, TurnContext
from .intent import classify_intent, route_intent
from .model import BudgetedModel
from .budget import BudgetExceeded
from .retrieval import gate_result, CitationRegistry, retrieve_for_turn
from .agent import build_agent_graph, persist

LOGGER = logging.getLogger(__name__)


def make_turn_input(question, conversation_id, user_id, turn_id, messages):
    return {"raw_question": question, "question": question, "conversation_id": conversation_id,
            "user_id": user_id, "turn_id": turn_id, "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages],
            "evidence": [], "citations": [], "actions": [], "decision": {}, "final_text": "",
            "stop_reason": "", "budget_usage": {}, "tool_steps": 0, "node_trace": [],
            "intent": "", "route": "", "gate_allowed": False, "gate_reason": ""}


def build_workflow(deps=None, checkpointer=None, gate_threshold=None):
    from app.config import settings
    threshold = gate_threshold if gate_threshold is not None else settings.workflow_gate_threshold

    def traced(state, node):
        get_stream_writer()(("workflow_status", {"node": node}))
        return {"node_trace": state.get("node_trace", []) + [node]}

    def resolve(state: WorkflowState):
        return {**traced(state, "resolve"), "question": state["raw_question"]}

    async def intent(state: WorkflowState, runtime: Runtime[TurnContext]):
        context = runtime.context
        update = traced(state, "intent")
        try:
            decision = await classify_intent(BudgetedModel(context.agent.model, context.budget, 128), state["question"])
            update.update(intent=decision.intent, route=route_intent(decision.intent))
        except (ValueError, BudgetExceeded) as exc:
            update.update(route="fallback", gate_reason="intent_unavailable", stop_reason=str(exc))
        update["budget_usage"] = context.budget.snapshot()
        return update

    async def retrieval(state: WorkflowState, runtime: Runtime[TurnContext]):
        context = runtime.context
        update = traced(state, "retrieval")
        try:
            result = await (context.retrieve or retrieve_for_turn)(state["question"], context.budget)
            gate = gate_result(result, threshold)
            registry = CitationRegistry()
            update.update(evidence=registry.merge(result.hits) if gate.allowed else [],
                          citations=registry.citations if gate.allowed else [], gate_allowed=gate.allowed, gate_reason=gate.reason)
        except Exception as exc:
            LOGGER.exception("Retrieval unavailable conversation_id=%s", state["conversation_id"])
            update.update(gate_allowed=False, gate_reason="retrieval_unavailable:" + type(exc).__name__)
        update["budget_usage"] = context.budget.snapshot()
        return update

    async def confidence_gate(state: WorkflowState, runtime: Runtime[TurnContext]):
        update = traced(state, "confidence_gate")
        if not state["gate_allowed"] and runtime.context.record_low_confidence is not None:
            try:
                await runtime.context.record_low_confidence(state["raw_question"], state["gate_reason"])
            except Exception:
                LOGGER.exception("low-confidence persistence failed conversation_id=%s", state["conversation_id"])
        return update

    def fixed(node, text, actions=None):
        async def handler(state: WorkflowState, runtime: Runtime[TurnContext]):
            update = traced(state, node)
            current = text
            if node == "fallback" and "unavailable" in state.get("gate_reason", ""):
                current = "查询服务暂时不可用，请稍后重试或选择人工核实。"
            message = AIMessage(content=current, id=f"{state['turn_id']}:final")
            await persist(runtime.context, message)
            get_stream_writer()(("delta", {"delta": current, "node": node}))
            update.update(messages=[message], final_text=current, actions=actions or [], stop_reason=node)
            return update
        return handler

    async def log(state: WorkflowState, runtime: Runtime[TurnContext]):
        update = traced(state, "log")
        if state.get("actions") and runtime.context.issue_actions is not None:
            actions = await runtime.context.issue_actions(state["actions"])
            update["actions"] = actions
            get_stream_writer()(("actions", {"actions": actions, "conversation_id": state["conversation_id"]}))
        update["budget_usage"] = runtime.context.budget.snapshot()
        LOGGER.info("workflow conversation_id=%s turn_id=%s intent=%s trace=%s stop=%s usage=%s",
                    state["conversation_id"], state["turn_id"], state.get("intent"), update["node_trace"],
                    state.get("stop_reason"), update["budget_usage"])
        get_stream_writer()(("workflow_summary", {"node_trace": update["node_trace"], "stop_reason": state.get("stop_reason"),
                                                   "usage": update["budget_usage"]}))
        return update

    builder = StateGraph(WorkflowState, context_schema=TurnContext)
    nodes = {"resolve": resolve, "intent": intent, "retrieval": retrieval, "confidence_gate": confidence_gate,
             "agent": build_agent_graph(deps), "fallback": fixed("fallback", "知识库证据不足，暂时无法确认，请补充信息或人工核实。"),
             "complaint": fixed("complaint", "抱歉给您带来不好的体验。您可以分别选择转人工或创建工单，也可以继续告诉我具体问题。", ["handoff", "ticket"]),
             "smalltalk": fixed("smalltalk", "您好，我是小喵，商品、订单、物流和售后问题都可以问我。"), "log": log}
    for name, function in nodes.items(): builder.add_node(name, function)
    builder.add_edge(START, "resolve")
    builder.add_edge("resolve", "intent")
    builder.add_conditional_edges("intent", lambda s: {"knowledge": "retrieval", "business": "agent", "complaint": "complaint", "smalltalk": "smalltalk", "fallback": "fallback"}[s["route"]])
    builder.add_edge("retrieval", "confidence_gate")
    builder.add_conditional_edges("confidence_gate", lambda s: "agent" if s["gate_allowed"] else "fallback")
    for node in ["agent", "fallback", "complaint", "smalltalk"]: builder.add_edge(node, "log")
    builder.add_edge("log", END)
    return builder.compile(checkpointer=checkpointer)
