"""Run the bare loop; --offline exercises real mock tools without API fees."""
import argparse
import asyncio
from langchain_core.messages import AIMessage, HumanMessage
from app.workflow.bare_loop import run_bare_loop
from app.workflow.budget import Limits, TurnBudget
from app.workflow.contracts import AgentDependencies
from app.services.tool_executor import ToolExecutor
from app.tools.business import build_tools


class DemoModel:
    def __init__(self):
        self.step = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages, **kwargs):
        self.step += 1
        if self.step <= 2:
            name = "query_order" if self.step == 1 else "query_logistics"
            return AIMessage(content="", tool_calls=[{
                "name": name, "args": {"order_id": "1001"}, "id": f"demo-{self.step}"}])
        return AIMessage(content=f"物流工具返回：{messages[-1].content}")


async def main(offline):
    from app.config import settings
    if offline:
        model = DemoModel()
    else:
        from app.core.llm import get_chat_model
        model = get_chat_model(disable_thinking=True).model_copy(update={"max_retries": 0})
    tools = build_tools(lambda: None, 1)
    # The demonstration only advertises read-only mock tools, not FAQ/database actions.
    tools = [tool for tool in tools if tool.name in {"query_order", "query_product", "query_logistics"}]
    budget = TurnBudget(Limits(model_calls=settings.agent_max_model_calls,
                              tools=settings.agent_max_tool_steps,
                              tokens=settings.agent_max_tokens,
                              seconds=settings.agent_timeout_seconds))
    result = await run_bare_loop([HumanMessage(content="先查订单1001，再查物流")],
                                AgentDependencies(model, tools, ToolExecutor(tools)), budget)
    for message in result.messages:
        print(message.type, message.content or getattr(message, "tool_calls", []))
    print("stop_reason:", result.stop_reason, "usage:", budget.snapshot())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    asyncio.run(main(parser.parse_args().offline))
