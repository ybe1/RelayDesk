async def test_official_sqlite_saver_survives_reopen(tmp_path):
    from app.workflow.runtime import checkpoint_runtime
    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict
    class State(TypedDict): value: int
    builder = StateGraph(State)
    builder.add_node("increment", lambda state: {"value": state["value"]+1})
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    config = {"configurable": {"thread_id": "trusted-conversation-1"}}
    path = tmp_path / "checkpoints.sqlite"
    async with checkpoint_runtime(path) as saver:
        graph = builder.compile(checkpointer=saver)
        assert (await graph.ainvoke({"value": 4}, config))["value"] == 5
    async with checkpoint_runtime(path) as saver:
        graph = builder.compile(checkpointer=saver)
        assert (await graph.aget_state(config)).values["value"] == 5


async def test_failed_node_keeps_prior_checkpoint_without_automatic_replay(tmp_path):
    import pytest
    from app.workflow.runtime import checkpoint_runtime
    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict
    class State(TypedDict): value: int
    calls = []
    def fail(state):
        calls.append(1)
        raise ConnectionError("upstream")
    builder = StateGraph(State)
    builder.add_node("before", lambda state: {"value": 10})
    builder.add_node("failed", fail)
    builder.add_edge(START, "before")
    builder.add_edge("before", "failed")
    builder.add_edge("failed", END)
    config = {"configurable": {"thread_id": "failed-turn"}}
    path = tmp_path / "failure.sqlite"
    async with checkpoint_runtime(path) as saver:
        graph = builder.compile(checkpointer=saver)
        with pytest.raises(ConnectionError): await graph.ainvoke({"value": 1}, config)
    async with checkpoint_runtime(path) as saver:
        graph = builder.compile(checkpointer=saver)
        assert (await graph.aget_state(config)).values["value"] == 10
        assert len(calls) == 1
