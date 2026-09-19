import asyncio
import json

from langchain.tools import tool
from pydantic import BaseModel

from app.services.tool_executor import ToolExecutor


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


def test_read_only_transient_failure_retries_twice():
    attempts = 0

    @tool(args_schema=Input)
    def query_faq(value: str) -> str:
        """Find a FAQ."""
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("temporary")
        return value

    result = asyncio.run(ToolExecutor([query_faq], backoff_seconds=0).execute(
        {"name": "query_faq", "args": {"value": "policy"}, "id": "call-4"}
    ))
    assert attempts == 3
    assert json.loads(result.content)["result"] == "policy"


def test_ticket_write_failure_is_never_retried():
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
    assert attempts == 1
    assert json.loads(result.content)["error"]["code"] == "execution_failed"


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
