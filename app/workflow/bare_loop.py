"""The bare tool loop: model -> tools -> observations -> model."""
import asyncio
import json
from langchain_core.messages import AIMessage, ToolMessage
from .budget import BudgetExceeded
from .contracts import AgentResult
from .model import BudgetedModel


def tool_error(call, code):
    return ToolMessage(tool_call_id=call["id"], content=json.dumps(
        {"error": {"code": code, "message": "本次工具未执行"}}, ensure_ascii=False))


async def run_bare_loop(messages, deps, budget):
    messages = list(messages)
    model = BudgetedModel(deps.model.bind_tools(
        [tool for tool in deps.tools if tool.name != "create_ticket"]), budget)
    while True:
        try:
            response = await model.ainvoke(messages)
        except BudgetExceeded as exc:
            messages.append(AIMessage(content="本次查询已达到处理上限，请补充问题后重试。"))
            return AgentResult(messages, str(exc))
        messages.append(response)
        if not response.tool_calls:
            return AgentResult(messages, "answered")
        stopped = None
        for call in response.tool_calls:
            if stopped:
                messages.append(await deps.executor.fail(call,stopped,'本轮预算耗尽，工具未执行'))
            elif call["name"] == "create_ticket":
                messages.append(await deps.executor.execute(call))
            else:
                try:
                    async with asyncio.timeout(budget.remaining_seconds()):
                        observation = await deps.executor.execute(call, before_attempt=budget.reserve_tool)
                    messages.append(observation)
                except (BudgetExceeded, TimeoutError) as exc:
                    stopped = str(exc) if isinstance(exc, BudgetExceeded) else "deadline"
                    messages.append(tool_error(call, stopped))
        if stopped:
            messages.append(AIMessage(content="本次查询已达到处理上限，请补充问题后重试。"))
            return AgentResult(messages, stopped)
