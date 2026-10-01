import json
from langchain_core.messages import AIMessage, ToolMessage


class SequenceModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
    def bind_tools(self, tools, **kwargs):
        assert all(tool.name != "create_ticket" for tool in tools)
        return self
    async def ainvoke(self, messages, **kwargs):
        self.requests.append(list(messages))
        return next(self.responses)


async def bare_scenario(names, *, tool_limit=5):
    from app.workflow.bare_loop import run_bare_loop
    from app.workflow.contracts import AgentDependencies
    from app.workflow.budget import Limits, TurnBudget
    from app.services.tool_executor import ToolExecutor
    from app.tools.business import build_tools
    from langchain_core.messages import HumanMessage
    responses = [AIMessage(content="", tool_calls=[{"name": name, "args": {"order_id": "1001"}, "id": str(i)}])
                 for i, name in enumerate(names)]
    responses.append(AIMessage(content="演示物流结果"))
    model = SequenceModel(responses)
    tools = build_tools(lambda: None, 1)
    executor = ToolExecutor(tools)
    deps = AgentDependencies(model, tools, executor)
    result = await run_bare_loop([HumanMessage(content="先查订单1001，再查物流")], deps,
                                 TurnBudget(Limits(tools=tool_limit), clock=lambda: 0))
    if len(model.requests) > 1:
        assert isinstance(model.requests[1][-1], ToolMessage)
        assert json.loads(model.requests[1][-1].content)["result"]
    calls = [call["name"] for message in result.messages if isinstance(message, AIMessage)
             for call in message.tool_calls]
    return result, calls
