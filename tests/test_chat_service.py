import asyncio

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from app.schemas.chat import ChatRequest


class MemoryStore:
    def __init__(self):
        self.rows = []

    def start_or_load(self, user_id, conversation_id):
        return conversation_id or 7

    def append(self, conversation_id, message):
        self.rows.append(message)

    def load_history(self, conversation_id):
        return list(self.rows)


class Model:
    def __init__(self, selection):
        self.selection = selection
        self.bind_count = 0

    def bind_tools(self, tools, tool_choice):
        assert tool_choice == "auto"
        self.bind_count += 1
        return self

    async def ainvoke(self, messages):
        return self.selection

    async def astream(self, messages):
        for char in "查到了":
            yield AIMessageChunk(content=char)


def collect(service, req):
    async def run():
        return [event async for event in service.events(req)]

    return asyncio.run(run())


def test_no_tool_still_streams_and_persists_final_answer():
    from app.services.chat import ChatService
    store = MemoryStore()
    model = Model(AIMessage(content="我来回答"))
    service = ChatService(model, store, lambda cid: [], lambda tools: None)
    events = collect(service, ChatRequest(user_id="u1", message="你好"))
    assert events[0] == ("conversation", {"conversation_id": 7})
    assert "".join(data["delta"] for kind, data in events if kind == "delta") == "查到了"
    assert [m.content for m in store.rows] == ["你好", "查到了"]
    assert events[-1] == ("done", None)


def test_multiple_requests_execute_only_first_and_match_all_tool_ids():
    from app.services.chat import ChatService
    store = MemoryStore()
    selection = AIMessage(content="", tool_calls=[
        {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "call-1"},
        {"name": "query_faq", "args": {"keyword": "邮费"}, "id": "call-2"},
    ])
    model = Model(selection)
    executed = []

    class Executor:
        async def execute(self, call):
            executed.append(call["name"])
            return ToolMessage(content='{"mock":true}', tool_call_id=call["id"])

    service = ChatService(model, store, lambda cid: [object()], lambda tools: Executor())
    events = collect(service, ChatRequest(user_id="u1", conversation_id=7, message="订单物流"))
    assert executed == ["query_logistics"]
    assert [m.tool_call_id for m in store.rows if isinstance(m, ToolMessage)] == ["call-1", "call-2"]
    assert [m.type for m in store.rows] == ["human", "ai", "tool", "tool", "ai"]
    assert any(kind == "tool_status" and data["tool_name"] == "query_logistics" for kind, data in events)
    assert events[-1] == ("done", None)


def test_tool_success_status_is_not_confused_by_result_text():
    from app.services.chat import ChatService

    selection = AIMessage(content="", tool_calls=[
        {"name": "query_faq", "args": {"keyword": "错误"}, "id": "call-3"},
    ])

    class Executor:
        async def execute(self, call):
            return ToolMessage(content='{"result":"error"}', tool_call_id=call["id"])

    service = ChatService(Model(selection), MemoryStore(), lambda cid: [object()], lambda tools: Executor())
    events = collect(service, ChatRequest(user_id="u1", message="错误"))
    phases = [data["phase"] for kind, data in events if kind == "tool_status"]
    assert phases == ["running", "success"]


def test_disconnect_after_running_status_persists_matching_tool_result():
    from app.services.chat import ChatService

    selection = AIMessage(content="", tool_calls=[
        {"name": "query_faq", "args": {"keyword": "退货"}, "id": "call-4"},
    ])
    store = MemoryStore()
    service = ChatService(Model(selection), store, lambda cid: [object()], lambda tools: None)

    async def disconnect():
        stream = service.events(ChatRequest(user_id="u1", message="退货"))
        assert (await anext(stream))[0] == "conversation"
        assert (await anext(stream))[0] == "tool_status"
        await stream.aclose()

    asyncio.run(disconnect())
    results = [m for m in store.rows if isinstance(m, ToolMessage)]
    assert len(results) == 1
    assert results[0].tool_call_id == "call-4"
    assert "interrupted" in results[0].content


def test_disconnect_after_success_status_keeps_completed_tool_result():
    from app.services.chat import ChatService

    selection = AIMessage(content="", tool_calls=[
        {"name": "query_faq", "args": {"keyword": "退货"}, "id": "call-5"},
    ])
    store = MemoryStore()

    class Executor:
        async def execute(self, call):
            return ToolMessage(content='{"result":"ok"}', tool_call_id=call["id"])

    service = ChatService(Model(selection), store, lambda cid: [object()], lambda tools: Executor())

    async def disconnect():
        stream = service.events(ChatRequest(user_id="u1", message="退货"))
        assert (await anext(stream))[0] == "conversation"
        assert (await anext(stream))[1]["phase"] == "running"
        assert (await anext(stream))[1]["phase"] == "success"
        await stream.aclose()

    asyncio.run(disconnect())
    results = [m for m in store.rows if isinstance(m, ToolMessage)]
    assert len(results) == 1
    assert results[0].tool_call_id == "call-5"
    assert "ok" in results[0].content
