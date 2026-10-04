import asyncio
import json

import pytest

from langchain.tools import tool
from pydantic import BaseModel

from app.services.tool_executor import ToolExecutor as RealToolExecutor, POLICY_PATH
from app.tooling.policy import PolicyStore
from app.tooling.contracts import ToolPolicy


class FixturePolicy(PolicyStore):
    def lookup(self,key):
        # These two are explicit test registrations, not production permissions.
        if key in {('builtin',None,'echo'),('builtin',None,'query_logistics')}:
            return ToolPolicy(max_retries=2 if key[-1]=='query_logistics' else 0)
        return super().lookup(key)


def ToolExecutor(tools,**kwargs):
    return RealToolExecutor(tools,policy=FixturePolicy(POLICY_PATH),**kwargs)


class Input(BaseModel):
    value: str


@tool(args_schema=Input)
def echo(value: str) -> str:
    """Echo a value."""
    return value


def test_executor_matches_call_id_and_validates_arguments():
    executor = ToolExecutor([echo])
    ok = asyncio.run(executor.execute({"name": "echo", "args": {"value": "hi"}, "id": "call-1"}))
    assert ok.tool_call_id == "call-1"
    assert json.loads(ok.content)["result"] == "hi"
    bad = asyncio.run(executor.execute({"name": "echo", "args": {}, "id": "call-2"}))
    assert json.loads(bad.content)["error"]["code"] == "invalid_arguments"
    unknown = asyncio.run(executor.execute({"name": "missing", "args": {}, "id": "call-3"}))
    assert json.loads(unknown.content)["error"]["code"] == "unknown_tool"


@pytest.mark.parametrize("name", ["query_order", "query_product", "query_logistics"])
def test_read_only_transient_failure_retries_twice(name):
    attempts = 0

    @tool(name, args_schema=Input)
    def query_order(value: str) -> str:
        """Look up an order."""
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("temporary")
        return value

    result = asyncio.run(ToolExecutor([query_order], backoff_seconds=0).execute(
        {"name": name, "args": {"value": "policy"}, "id": "call-4"}
    ))
    assert attempts == 3
    assert json.loads(result.content)["result"] == "policy"


def test_ticket_without_confirmation_is_not_executed():
    attempts = 0

    @tool(args_schema=Input)
    def create_ticket(value: str) -> str:
        """Create a ticket."""
        nonlocal attempts
        attempts += 1
        raise ConnectionError("uncertain commit")

    result = asyncio.run(ToolExecutor([create_ticket], backoff_seconds=0).execute(
        {"name": "create_ticket", "args": {"value": "help"}, "id": "call-5"}
    ))
    assert attempts == 0
    assert json.loads(result.content)["error"]["code"] == "permission_denied"


def test_read_only_timeout_is_bounded_and_returns_error():
    attempts = 0

    @tool(args_schema=Input)
    async def query_order(value: str) -> str:
        """Look up an order."""
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(0.05)
        return value

    result = asyncio.run(ToolExecutor([query_order], timeout_seconds=0.001, backoff_seconds=0).execute(
        {"name": "query_order", "args": {"value": "1001"}, "id": "call-6"}
    ))
    assert attempts == 3
    assert json.loads(result.content)["error"]["code"] == "timeout"


def test_oversized_tool_result_returns_bounded_operational_error():
    result = asyncio.run(ToolExecutor([echo]).execute({"name": "echo", "args": {"value": "😀" * 60000}, "id": "large"}))
    assert len(result.content.encode("utf-8")) <= 60000
    assert json.loads(result.content)["error"]["code"] == "execution_failed"


def test_default_faq_budget_allows_multistage_retrieval_longer_than_five_seconds():
    @tool(args_schema=Input)
    async def query_faq(value: str) -> str:
        """Run a multistage FAQ lookup."""
        await asyncio.sleep(5.05)
        return value
    result = asyncio.run(ToolExecutor([query_faq], backoff_seconds=0).execute({"name": "query_faq", "args": {"value": "ok"}, "id": "slow"}))
    assert json.loads(result.content) == {"result": "ok"}
