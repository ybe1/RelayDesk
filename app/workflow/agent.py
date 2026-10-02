"""Bounded ReAct subgraph; decisions remain private, only final output streams."""
import asyncio
import json
from langchain_core.messages import AIMessage, SystemMessage,ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langgraph.config import get_stream_writer
from .contracts import WorkflowState, TurnContext, AgentDecision
from .budget import BudgetExceeded
from .model import BudgetedModel
from .bare_loop import tool_error
from .retrieval import CitationRegistry
from app.context.compile import compile_agent_messages,AGENT_SYSTEM,project_evidence
from app.context.contracts import HistoryProjection,LayerAnchors,ContextConfig
from app.context.budget import calculate_budget
from app.context.logging import record_model
from app.context.tokens import estimate_text

STOP_TEXT = "本次查询已达到处理上限，请补充问题后重试。"
async def persist(context, message):
    if context.save_message is not None:
        await context.save_message(message)

def _messages(state,context,phase):
    cfg=context.context_config or ContextConfig()
    if context.history_projection is not None:
        projection=context.history_projection
        current=[m for m in state['messages'] if (m.id or '').startswith(state['turn_id']+':') and m.type!='human']
        raw=state['raw_question']
    else:
        messages=state['messages']
        start=max((i for i,m in enumerate(messages) if m.type=='human'),default=0)
        projection=HistoryProjection(LayerAnchors(),tuple(messages[:start]),(),'',0,())
        raw=messages[start].content if messages else state.get('raw_question','')
        current=messages[start+1:]
    return compile_agent_messages(projection,raw,state.get('evidence',[]),state.get('order_snapshot',{}),
        current,phase,tool_limit=cfg.tool_result,assessment=state.get('refund_assessment'),evidence_limit=cfg.evidence_per_hit)

def _model(selected,context,state,phase,tools):
    cfg=context.context_config or ContextConfig()
    wrapped=BudgetedModel(selected.model.bind_tools(tools,tool_choice='none' if phase!='decision' else 'auto'),
        context.budget,output_cap=384 if phase=='decision' else cfg.output)
    wrapped.context_logger=lambda messages,bound:record_model(state.get('conversation_id'),state['turn_id'],phase,
        messages,bound,calculate_budget(cfg,estimate_text(context.history_projection.summary) if context.history_projection else 0))
    return wrapped


def build_agent_graph(deps=None):
    async def decision(state: WorkflowState, runtime: Runtime[TurnContext]):
        context = runtime.context
        selected = deps or context.agent
        read_tools = [t for t in selected.tools if t.name != "create_ticket"]
        messages = _messages(state,context,'decision')
        try:
            rows,complete=project_evidence(state.get('evidence',[]))
            if not complete:
                raise ValueError('关键政策条款超出上下文额度，需要人工核实')
            raw = await _model(selected,context,state,'decision',[*read_tools,AgentDecision]).ainvoke(messages)
            finish = [call for call in raw.tool_calls if call["name"] == "AgentDecision"]
            if finish:
                if len(raw.tool_calls) != 1:
                    raise ValueError("finish decision cannot mix with business calls")
                choice = AgentDecision.model_validate(finish[0]["args"])
            elif raw.tool_calls:
                choice = AgentDecision(kind="tools", tool_calls=[{key: call[key] for key in ("name", "args", "id")}
                                                                  for call in raw.tool_calls], reason="模型选择业务工具")
            else:
                choice = AgentDecision(kind="answer", reason="模型未申请工具")
            if choice.kind == "tools" and not choice.tool_calls:
                raise ValueError("empty tools decision")
            if choice.kind != "tools" and choice.tool_calls:
                raise ValueError("non-tool decision contains tool calls")
            if 'refund' in choice.actions and (state.get('intent')!='退款退货'
                    or not choice.refund_assessment or choice.refund_assessment.status!='eligible'):
                choice=choice.model_copy(update={'actions':[action for action in choice.actions if action!='refund']})
            update = {"decision": choice.model_dump(), "actions": list(dict.fromkeys(choice.actions)),
                      "refund_assessment":choice.refund_assessment.model_dump() if choice.refund_assessment else {},
                      "node_trace": state.get("node_trace", []) + ["decision"]}
            if choice.kind == "tools":
                calls = [call.model_dump() for call in choice.tool_calls]
                # Ignore model-selected ids for storage identity; allocate stable per-step ids.
                for i, call in enumerate(calls):
                    call["id"] = f"{state['turn_id'][:24]}-{context.budget.model_calls}-{i}"
                message = AIMessage(content="", tool_calls=calls, id=f"{state['turn_id']}:decision:{context.budget.model_calls}")
                await persist(context, message)
                update["messages"] = [message]
        except (BudgetExceeded, ValueError) as exc:
            update = {"decision": {"kind": "stop"}, "stop_reason": str(exc),
                      "node_trace": state.get("node_trace", []) + ["decision"]}
        update["budget_usage"] = context.budget.snapshot()
        return update

    async def tools(state: WorkflowState, runtime: Runtime[TurnContext]):
        context = runtime.context
        selected = deps or context.agent
        messages, stopped = [], None
        registry = CitationRegistry(state.get("citations"))
        evidence = list(state.get("evidence", []))
        for call in state["messages"][-1].tool_calls:
            if stopped:
                observation = tool_error(call, stopped)
            elif call["name"] == "create_ticket":
                observation = tool_error(call, "user_confirmation_required")
            elif call['name']=='query_faq' and state.get('gate_allowed') and evidence:
                # Close any stale/model-invented request without retrieving
                # twice or duplicating the bulky evidence in message history.
                observation=ToolMessage(content=json.dumps({'result':json.dumps({
                    'sufficient':True,'reuse_existing_evidence':True,
                    'message':'本轮政策已检索并通过置信度闸，请使用本轮资料中的证据回答。'},ensure_ascii=False)},ensure_ascii=False),tool_call_id=call['id'])
            elif (call['name']=='query_order' and state.get('order_snapshot')
                  and call['args'].get('order_id')==state['order_snapshot'].get('order_id')):
                observation=ToolMessage(content=json.dumps({'result':json.dumps(state['order_snapshot'],ensure_ascii=False)},ensure_ascii=False),tool_call_id=call['id'])
            else:
                try:
                    get_stream_writer()(("tool_status", {"tool": call["name"], "status": "running", "tool_call_id": call["id"]}))
                    async with asyncio.timeout(context.budget.remaining_seconds()):
                        observation = await selected.executor.execute(call, before_attempt=context.budget.reserve_tool)
                except (BudgetExceeded, TimeoutError) as exc:
                    stopped = str(exc) if isinstance(exc, BudgetExceeded) else "deadline"
                    observation = tool_error(call, stopped)
                if call["name"] == "query_faq":
                    try:
                        payload = json.loads(observation.content)
                        faq = json.loads(payload["result"])
                        if faq.get("sufficient"):
                            mapped = registry.merge(faq["evidence"])
                            faq["evidence"], faq["citations"] = mapped, registry.citations
                            evidence.extend(mapped)
                            observation.content = json.dumps({"result": json.dumps(faq, ensure_ascii=False)}, ensure_ascii=False)
                    except (KeyError, ValueError, TypeError):
                        pass
            observation.id = f"{state['turn_id']}:tool:{call['id']}"
            await persist(context, observation)
            messages.append(observation)
            get_stream_writer()(("tool_status", {"tool_name": call["name"], "phase": "failed" if "error" in json.loads(observation.content) else "success", "tool_call_id": call["id"]}))
        return {"messages": messages, "stop_reason": stopped or "", "budget_usage": context.budget.snapshot(),
                "tool_steps": context.budget.tool_steps, "citations": registry.citations, "evidence": evidence,
                "node_trace": state.get("node_trace", []) + ["tools"]}

    async def final_answer(state: WorkflowState, runtime: Runtime[TurnContext]):
        context = runtime.context
        selected = deps or context.agent
        writer = get_stream_writer()
        if state.get("citations"):
            writer(("citations", {"citations": state["citations"]}))
        mode = state["decision"]["kind"]
        messages = _messages(state,context,'clarify' if mode=='clarify' else 'final_answer')
        content, reason = "", "clarify" if mode == "clarify" else "answered"
        try:
            read_tools=[t for t in selected.tools if t.name!='create_ticket']
            async for chunk in _model(selected,context,state,'final_answer',[*read_tools,AgentDecision]).astream(messages):
                if isinstance(chunk.content, str) and chunk.content:
                    content += chunk.content
                    writer(("delta", {"delta": chunk.content, "node": "final_answer"}))
        except BudgetExceeded as exc:
            reason = str(exc)
            content += STOP_TEXT
            writer(("delta", {"delta": STOP_TEXT, "node": "final_answer"}))
        message = AIMessage(content=content, id=f"{state['turn_id']}:final",
            additional_kwargs={'citations':state.get('citations',[])})
        await persist(context, message)
        return {"messages": [message], "final_text": content, "stop_reason": reason,
                "budget_usage": context.budget.snapshot(), "node_trace": state.get("node_trace", []) + ["final_answer"]}

    async def stop(state: WorkflowState, runtime: Runtime[TurnContext]):
        text='政策条款超出当前上下文额度，暂时无法确认，请选择人工核实。' if '关键政策' in state.get('stop_reason','') else STOP_TEXT
        message = AIMessage(content=text, id=f"{state['turn_id']}:final")
        await persist(runtime.context, message)
        get_stream_writer()(("delta", {"delta": text, "node": "stop"}))
        return {"messages": [message], "final_text": text, "budget_usage": runtime.context.budget.snapshot(),
                "node_trace": state.get("node_trace", []) + ["stop"]}

    builder = StateGraph(WorkflowState, context_schema=TurnContext)
    for name, function in [("decision", decision), ("tools", tools), ("final_answer", final_answer), ("stop", stop)]:
        builder.add_node(name, function)
    builder.add_edge(START, "decision")
    builder.add_conditional_edges("decision", lambda s: {"tools": "tools", "answer": "final_answer", "clarify": "final_answer", "stop": "stop"}[s["decision"]["kind"]])
    builder.add_conditional_edges("tools", lambda s: "stop" if s.get("stop_reason") else "decision")
    builder.add_edge("final_answer", END)
    builder.add_edge("stop", END)
    return builder.compile()
