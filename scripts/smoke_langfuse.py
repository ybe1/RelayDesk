"""Create and read back one synthetic trace from the local Langfuse stack."""

import asyncio
import os
import time
from typing import TypedDict

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph


class SmokeState(TypedDict):
    value: int


async def _emit_graph() -> str:
    builder = StateGraph(SmokeState)
    names = ("resolve", "intent", "retrieval", "confidence_gate", "agent")
    for name in names:
        builder.add_node(name, lambda state: {"value": state["value"] + 1})
    builder.add_edge(START, names[0])
    for left, right in zip(names, names[1:]):
        builder.add_edge(left, right)
    builder.add_edge(names[-1], END)
    handler = CallbackHandler()
    await builder.compile().ainvoke({"value": 0}, {"callbacks": [handler], "metadata": {
        "langfuse_session_id": "ch09-local-smoke", "langfuse_tags": ["synthetic-smoke"]}})
    if not handler.last_trace_id:
        raise RuntimeError("LangGraph callback did not produce a trace id")
    return handler.last_trace_id


def main() -> int:
    client = Langfuse(host=os.environ.get("LANGFUSE_HOST", "http://127.0.0.1:3000"))
    trace_id = asyncio.run(_emit_graph())
    client.flush()
    last_error = None
    last_names: set[str | None] = set()
    # Langfuse documents a 15-30 second server-side ingestion lag after flush.
    for _ in range(60):
        try:
            observations = client.api.observations.get_many(trace_id=trace_id, limit=10)
            rows = observations.data
            names = {row.name for row in rows}
            last_names = names
            expected = {"resolve", "intent", "retrieval", "confidence_gate", "agent"}
            if expected <= names:
                print({"trace_id": trace_id, "observations": len(rows), "nodes": sorted(expected)})
                client.shutdown()
                return 0
        except Exception as exc:
            last_error = exc
            time.sleep(0.5)
    client.shutdown()
    raise RuntimeError(
        "synthetic trace was not readable after export: "
        f"error={last_error}, observed_names={sorted(str(name) for name in last_names)}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
