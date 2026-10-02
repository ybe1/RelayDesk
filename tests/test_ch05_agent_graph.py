import json
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from app.workflow.budget import TurnBudget, Limits
from app.workflow.contracts import AgentDependencies
from app.tools.business import build_tools
from app.services.tool_executor import ToolExecutor


class GraphModel:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.calls = 0
        self.streams = 0
        self.advertised = []
    def with_structured_output(self, schema, **kwargs): return self
    def bind_tools(self, tools, **kwargs):
        self.advertised = [getattr(t, "name", getattr(t, "__name__", "")) for t in tools]
        self.bound = True
        return self
    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        decision = next(self.decisions)
        if isinstance(decision, dict) and getattr(self, "bound", False):
            calls = decision["tool_calls"] if decision["kind"] == "tools" else [{"name": "AgentDecision", "args": decision, "id": "finish"}]
            return AIMessage(content="", tool_calls=calls, usage_metadata={"input_tokens": 100, "output_tokens": 30, "total_tokens": 130})
        return decision
    async def astream(self, messages, **kwargs):
        self.streams += 1
        yield AIMessageChunk(content="演示")
        yield AIMessageChunk(content="物流结果", usage_metadata={"input_tokens": 100, "output_tokens": 5, "total_tokens": 105})


def tools_decision(name, id="call"):
    return {"kind": "tools", "tool_calls": [{"name": name, "args": {"order_id": "1001"}, "id": id}], "actions": [], "reason": "需查询"}


async def test_pretrieved_policy_cannot_repeat_external_faq_search():
    from app.workflow.agent import build_agent_graph
    from app.workflow.contracts import TurnContext
    class CountingExecutor:
        calls=0
        async def execute(self,call,**kwargs):
            self.calls+=1
            return ToolMessage(content=json.dumps({'result':json.dumps({'sufficient':False})}),tool_call_id=call['id'])
    executor=CountingExecutor()
    model=GraphModel([{'kind':'tools','tool_calls':[{'name':'query_faq','args':{'keyword':'退货'},'id':'faq'}],'reason':'重复检索'},
        {'kind':'answer','actions':[],'reason':'证据足够'}])
    budget=TurnBudget(Limits(tokens=100000))
    context=TurnContext(agent=AgentDependencies(model,build_tools(lambda:None,1),executor),budget=budget)
    result=await build_agent_graph().ainvoke({'messages':[HumanMessage(content='退货政策是什么')],
        'turn_id':'cached-faq','gate_allowed':True,'node_trace':[],
        'evidence':[{'n':1,'id':1,'answer':'商品完好可退'}],'citations':[]},context=context)
    assert executor.calls==0 and budget.tool_steps==0
    assert 'query_faq' not in model.advertised
    assert result['stop_reason']=='answered'


async def scenario(decisions, limits=None):
    from app.workflow.agent import build_agent_graph
    from app.workflow.contracts import TurnContext
    model = GraphModel(decisions)
    tools = build_tools(lambda: None, 1)
    deps = AgentDependencies(model, tools, ToolExecutor(tools))
    budget = TurnBudget(limits or Limits(tokens=100000))
    context = TurnContext(agent=deps, budget=budget)
    graph = build_agent_graph()
    events = []
    async for event in graph.astream({"messages": [HumanMessage(content="订单1001物流")],
            "turn_id": "test", "node_trace": [], "citations": [], "evidence": [], "actions": []},
            context=context, stream_mode=["custom", "updates"]):
        events.append(event)
    return events, model, budget


async def test_final_stream_hides_decisions_and_multi_step_tools():
    events, model, budget = await scenario([tools_decision("query_order", "a"), tools_decision("query_logistics", "b"),
        {"kind": "answer", "tool_calls": [], "actions": [], "reason": "足够"}])
    deltas = [v for mode, v in events if mode == "custom" and v[0] == "delta"]
    assert [v[1]["delta"] for v in deltas] == ["演示", "物流结果"]
    assert all(v[1]["node"] == "final_answer" for v in deltas)
    assert model.calls == 3 and model.streams == 1
    assert budget.snapshot()["tool_steps"] == 2


async def test_agent_advertises_real_function_tools_and_finish_schema():
    _, model, _ = await scenario([{"kind": "answer", "tool_calls": [], "actions": [], "reason": "足够"}])
    assert "query_logistics" in model.advertised
    assert "AgentDecision" in model.advertised
    assert "create_ticket" not in model.advertised


async def test_tool_budget_stops_without_extra_generation():
    events, model, budget = await scenario([tools_decision("query_order")], Limits(tools=0, tokens=100000))
    assert model.calls == 1 and model.streams == 0
    updates = [v for mode, v in events if mode == "updates"]
    results = [m for v in updates for d in v.values() for m in d.get("messages", []) if isinstance(m, ToolMessage)]
    assert json.loads(results[0].content)["error"]["code"] == "tool_steps"


async def test_ticket_is_not_executed_and_clarify_streams_only_final_node():
    events, model, budget = await scenario([tools_decision("create_ticket"),
        {"kind": "clarify", "tool_calls": [], "actions": ["ticket"], "reason": "缺订单号"}])
    results = [m for mode, v in events if mode == "updates" for d in v.values()
               for m in d.get("messages", []) if isinstance(m, ToolMessage)]
    assert json.loads(results[0].content)["error"]["code"] == "user_confirmation_required"
    assert budget.snapshot()["tool_steps"] == 0


async def test_unknown_tool_returns_observation_and_model_limit_ends_loop():
    events, model, budget = await scenario([tools_decision("unknown")], Limits(model_calls=1, tokens=100000))
    results = [m for mode, v in events if mode == "updates" for d in v.values()
               for m in d.get("messages", []) if isinstance(m, ToolMessage)]
    assert json.loads(results[0].content)["error"]["code"] == "unknown_tool"
    assert model.calls == 1 and model.streams == 0
