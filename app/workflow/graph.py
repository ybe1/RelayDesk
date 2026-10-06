"""Deterministic parent workflow; model labels cannot override routing."""
import asyncio
import logging
from langchain_core.messages import AIMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime
from .contracts import WorkflowState, TurnContext
from .intent import classify_with_escalation, route_intent,detect_smalltalk
from .resolution import resolve_question
from .after_sales import build_after_sales_graph,refund_offer_defaults
from .routing_contracts import Resolution
from .model import BudgetedModel
from .budget import BudgetExceeded
from .retrieval import gate_result, calibrated_gate, CitationRegistry, retrieve_for_turn, check_for_turn
from .agent import build_agent_graph, persist
from app.context.compile import understanding_history
from .history import is_history_recall

LOGGER = logging.getLogger(__name__)


def make_turn_input(question, conversation_id, user_id, turn_id, messages):
    return {"raw_question": question, "question": question, "conversation_id": conversation_id,
            "user_id": user_id, "turn_id": turn_id, "messages": messages,
            "evidence": [], "evidence_snapshot": {}, "confidence": {}, "evidence_useful": None,
            "citations": [], "actions": [], "decision": {}, "final_text": "",
            "stop_reason": "", "budget_usage": {}, "tool_steps": 0, "node_trace": [],
            "intent": "", "route": "", "history_recall":False, "gate_allowed": False, "gate_reason": "",
            "resolution":{},"intent_confidence":0.0,"intent_model_tier":"","intent_upgraded":False,
            "order_id":None,"order_snapshot":{},"selection":{},"expanded_queries":[],"refund_assessment":{}}


def build_workflow(deps=None, checkpointer=None, gate_threshold=None):
    from app.config import settings
    threshold = gate_threshold if gate_threshold is not None else settings.workflow_gate_threshold

    def traced(state, node):
        get_stream_writer()(("workflow_status", {"node": node}))
        return {"node_trace": state.get("node_trace", []) + [node]}

    async def resolve(state: WorkflowState,runtime:Runtime[TurnContext]):
        context=runtime.context
        update=traced(state,'resolve')
        if state.get('ticket_request',{}).get('active'):
            return {**update,'question':state['raw_question'],'resolution':
                Resolution(question=state['raw_question'],reference_status='clear',scope='general',order_source='none').model_dump()}
        try:
            if detect_smalltalk(state['raw_question']) or is_history_recall(state['raw_question']):
                resolution=Resolution(question=state['raw_question'],reference_status='clear',scope='general',order_source='none')
            else:
                if context.history_projection is not None:
                    from app.context.compile import understanding_history
                    history=understanding_history(context.history_projection)
                else:
                    history=list(state['messages'])
                    if history and history[-1].type=='human' and history[-1].content==state['raw_question']: history=history[:-1]
                resolution=await resolve_question(context.agent.model,state['raw_question'],history,context.budget,
                    shared_history=context.history_projection is not None)
            update.update(question=resolution.question,resolution=resolution.model_dump(),order_id=resolution.order_id)
        except Exception as exc:
            update.update(resolution={},gate_reason='resolution_unavailable:'+type(exc).__name__)
        update['budget_usage']=context.budget.snapshot()
        return update

    async def intent(state: WorkflowState, runtime: Runtime[TurnContext]):
        context = runtime.context
        update = traced(state, "intent")
        if state.get('ticket_request',{}).get('active'):
            result={**update,'intent':'售后','route':'business','intent_confidence':1.0,
                'intent_model_tier':'user_action','budget_usage':context.budget.snapshot()}
            if context.observability: context.observability.tag_intent(result['intent'])
            return result
        if is_history_recall(state['raw_question']):
            result={**update,'intent':'订单','route':'recall','intent_confidence':1.0,
                'intent_model_tier':'rule','intent_upgraded':False,'budget_usage':context.budget.snapshot()}
            if context.observability: context.observability.tag_intent(result['intent'])
            return result
        try:
            if not state.get('resolution'): raise ValueError('resolution unavailable')
            run = await classify_with_escalation(context.agent.model,context.intent_small_model,state['question'],context.budget,
                                                settings.intent_confidence_threshold,
                                                history=understanding_history(context.history_projection) if context.history_projection is not None else None)
            decision=run.decision
            route=route_intent(decision.intent)
            if decision.intent in {'退款退货','售后'}:
                route='after_sales' if state['resolution']['scope']=='order_specific' else 'knowledge'
            update.update(intent=decision.intent,route=route,intent_confidence=decision.confidence,
                          intent_model_tier=run.model_tier,intent_upgraded=run.upgraded)
        except Exception as exc:
            update.update(route="fallback", gate_reason="intent_unavailable", stop_reason=str(exc))
        update["budget_usage"] = context.budget.snapshot()
        if context.observability and update.get('intent'):
            context.observability.tag_intent(update['intent'])
        return update

    async def retrieval(state: WorkflowState, runtime: Runtime[TurnContext]):
        context = runtime.context
        update = traced(state, "retrieval")
        try:
            if context.retrieve:
                result=await context.retrieve(state['question'],context.budget)
            else:
                from app.knowledge.query_understanding import QueryUnderstanding
                result=await retrieve_for_turn(state['question'],context.budget,
                    understood_query=QueryUnderstanding(normalized_query=state['question'],search_query=state['question']))
            if context.confidence_config is not None:
                snapshot, gate = calibrated_gate(state['question'], result, context.confidence_config)
                update.update(evidence_snapshot=snapshot.model_dump(mode='json'), confidence=gate.model_dump())
            else:
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
        operational = state.get("gate_reason", "").startswith(("retrieval_unavailable:", "expansion_unavailable:")) \
            or state.get("gate_reason") == "order_unavailable"
        if not state["gate_allowed"] and not operational and runtime.context.record_low_confidence is not None:
            try:
                callback = runtime.context.record_low_confidence
                try:
                    await callback(state["raw_question"], "retrieval_low_conf", state["gate_reason"],
                                   state.get("evidence_snapshot") or None, f"{state['turn_id']}:retrieval_low_conf")
                except TypeError:
                    await callback(state["raw_question"], state["gate_reason"])
            except Exception:
                LOGGER.exception("low-confidence persistence failed conversation_id=%s", state["conversation_id"])
        return update

    async def ensure_useful_evidence(state: WorkflowState, runtime: Runtime[TurnContext]):
        update = traced(state, "evidence_self_check")
        if runtime.context.confidence_config is None:
            return {**update, "evidence_useful": True}
        checker = runtime.context.evidence_check
        decision = (await checker(state["question"], state["evidence"], runtime.context.budget)
                    if checker else await check_for_turn(state["question"], state["evidence"],
                                                         runtime.context.agent.model, runtime.context.budget))
        update.update(evidence_useful=decision.useful, gate_reason=decision.reason)
        if not decision.useful and runtime.context.record_low_confidence is not None:
            try:
                await runtime.context.record_low_confidence(
                    state["raw_question"], "self_check", decision.reason,
                    state.get("evidence_snapshot") or None, f"{state['turn_id']}:self_check",
                )
            except Exception:
                LOGGER.exception("self-check persistence failed conversation_id=%s", state["conversation_id"])
        return update

    def fixed(node, text, actions=None):
        async def handler(state: WorkflowState, runtime: Runtime[TurnContext]):
            update = traced(state, node)
            current = text
            if node == "fallback" and "unavailable" in state.get("gate_reason", ""):
                current = "查询服务暂时不可用，请稍后重试或选择人工核实。"
            message = AIMessage(content=current, id=f"{state['turn_id']}:final",
                additional_kwargs={'evidence_snapshot':state.get('evidence_snapshot') or None,
                                   'gate_reason':state.get('gate_reason') or None})
            await persist(runtime.context, message)
            get_stream_writer()(("delta", {"delta": current, "node": node}))
            update.update(messages=[message], final_text=current, actions=actions or [], stop_reason=node)
            return update
        return handler

    async def log(state: WorkflowState, runtime: Runtime[TurnContext]):
        update = traced(state, "log")
        suggestions=[action for action in state.get('actions',[]) if action!='refund' or refund_offer_defaults(state) is not None]
        runtime.context.refund_defaults=refund_offer_defaults(state)
        if suggestions and runtime.context.issue_actions is not None:
            actions = await runtime.context.issue_actions(suggestions)
            update["actions"] = actions
            get_stream_writer()(("actions", {"actions": actions, "conversation_id": state["conversation_id"]}))
            final=next((m for m in reversed(state.get('messages',[])) if m.id==f"{state['turn_id']}:final"),None)
            if final is not None:
                update['messages']=[final.model_copy(update={'additional_kwargs':{**final.additional_kwargs,'actions':actions}})]
        update["budget_usage"] = runtime.context.budget.snapshot()
        if runtime.context.message_db_ids is not None:
            update['message_db_ids']=dict(runtime.context.message_db_ids)
        LOGGER.info("workflow conversation_id=%s turn_id=%s intent=%s trace=%s stop=%s usage=%s",
                    state["conversation_id"], state["turn_id"], state.get("intent"), update["node_trace"],
                    state.get("stop_reason"), update["budget_usage"])
        get_stream_writer()(("workflow_summary", {"node_trace": update["node_trace"], "stop_reason": state.get("stop_reason"),
                                                   "usage": update["budget_usage"],
                                                   "message_id":(runtime.context.message_db_ids or {}).get(f"{state['turn_id']}:final")}))
        return update

    def recall(state):
        return {**traced(state,'recall'),'history_recall':True,
            'decision':{'kind':'answer','tool_calls':[],'actions':[],'reason':'只回忆本会话记录，不查询当前业务数据'}}

    builder = StateGraph(WorkflowState, context_schema=TurnContext)
    nodes = {"resolve": resolve, "intent": intent, "retrieval": retrieval, "confidence_gate": confidence_gate,
             "evidence_self_check": ensure_useful_evidence,
             "after_sales":build_after_sales_graph(threshold),
             "agent": build_agent_graph(deps), 'recall':recall, "fallback": fixed("fallback", "知识库证据不足，暂时无法确认，请补充信息或人工核实。"),
             "complaint": fixed("complaint", "抱歉给您带来不好的体验。您可以分别选择转人工或创建工单，也可以继续告诉我具体问题。", ["handoff", "ticket"]),
             "smalltalk": fixed("smalltalk", "您好，我是小喵，商品、订单、物流和售后问题都可以问我。"), "log": log}
    for name, function in nodes.items(): builder.add_node(name, function)
    builder.add_edge(START, "resolve")
    builder.add_edge("resolve", "intent")
    builder.add_conditional_edges("intent", lambda s: {"knowledge": "retrieval", "business": "agent", 'recall':'recall', "after_sales":"after_sales", "complaint": "complaint", "smalltalk": "smalltalk", "fallback": "fallback"}[s["route"]])
    builder.add_edge('recall','agent')
    builder.add_edge('after_sales','confidence_gate')
    builder.add_edge("retrieval", "confidence_gate")
    builder.add_conditional_edges("confidence_gate", lambda s: "evidence_self_check" if s["gate_allowed"] else "fallback")
    builder.add_conditional_edges("evidence_self_check", lambda s: "agent" if s.get("evidence_useful") else "fallback")
    for node in ["agent", "fallback", "complaint", "smalltalk"]: builder.add_edge(node, "log")
    builder.add_edge("log", END)
    return builder.compile(checkpointer=checkpointer)
