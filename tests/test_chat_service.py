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

import json
import pytest


@pytest.mark.parametrize("outer", [None, [], "bad json", {"result": "not json"}, {"result": {"sufficient": False, "citations": [{"n": 1}]}}, {"error": {}, "result": {"sufficient": True, "citations": [{"n": 1}]}}])
def test_malformed_or_unsuccessful_tool_payload_never_emits_citations(outer):
    from app.services.chat import ChatService
    selection = AIMessage(content="", tool_calls=[{"name": "query_faq", "args": {"keyword": "退货"}, "id": "c"}])
    class Executor:
        async def execute(self, call):
            return ToolMessage(content=json.dumps(outer), tool_call_id="c")
    events = collect(ChatService(Model(selection), MemoryStore(), lambda _: [], lambda _: Executor()), ChatRequest(user_id="u", message="退货"))
    assert not any(kind == "citations" for kind, _ in events)
    assert events[-1] == ("done", None)


def test_citations_precede_delta_and_match_persisted_tool_message():
    from app.services.chat import ChatService
    from app.services.tool_executor import ToolExecutor
    from app.tools.business import build_tools
    selection = AIMessage(content="", tool_calls=[{"name": "query_faq", "args": {"keyword": "退货"}, "id": "c"}])
    store = MemoryStore()
    service = ChatService(Model(selection), store, lambda cid: build_tools(lambda: None, cid, lambda _: [{"id": 17, "question": "退货", "answer": "七天内", "category": "售后"}]), ToolExecutor)
    events = collect(service, ChatRequest(user_id="u", message="退货"))
    kinds = [kind for kind, _ in events]
    assert kinds.index("citations") < kinds.index("delta")
    citations = next(data["citations"] for kind, data in events if kind == "citations")
    message = next(m for m in store.rows if isinstance(m, ToolMessage))
    assert citations == json.loads(message.content)["result"]["citations"]
    assert citations[0]["chunk_id"] == 17


def test_chat_supplies_raw_question_without_changing_model_arguments():
    from app.services.chat import ChatService
    from app.services.tool_executor import ToolExecutor
    from app.tools.business import build_tools
    raw = "  我真急！邮费到底多少钱？  "
    store = MemoryStore()
    selection = AIMessage(content="", tool_calls=[{"name": "query_faq", "args": {"keyword": "邮费"}, "id": "c"}])
    received = []
    def factory(cid, *, raw_question=None):
        received.append(raw_question)
        return build_tools(lambda: None, cid, lambda _: [{"id": 17, "question": "邮费", "answer": "满99包邮", "category": "配送"}])
    events = collect(ChatService(Model(selection), store, factory, ToolExecutor), ChatRequest(user_id="u", message=raw))
    assert events[-1] == ("done", None)
    assert received == [raw]
    assert store.rows[1].tool_calls[0]["args"] == {"keyword": "邮费"}


@pytest.mark.parametrize("raw,keyword", [
    ("运费是多少？退货期限又是多少？", "运费"),
    ("不是 MH-V10 Pro，我问的是 MH-V10 的保修期限。", "MH-V10 Pro 保修期限"),
    ("它的功率是多少？", "MH-V10 Pro 功率"),
])
async def test_chat_query_understanding_only_receives_current_raw_question(structured_transport, raw, keyword):
    from app.knowledge.retrieval import RetrievalDependencies, search_knowledge_ch04
    from app.knowledge.rerank import RerankClient
    from app.services.chat import ChatService
    from app.services.tool_executor import ToolExecutor
    from app.tools.business import build_tools
    from tests.test_hybrid_retrieval import Store, Embedder, hit

    model, requests = structured_transport("QueryUnderstanding", {
        "normalized_query": raw, "search_query": raw, "category": None,
    })
    history = MemoryStore()
    history.rows = [HumanMessage(content="我想买 MH-V10 Pro"), AIMessage(content="请继续提问")]
    selection = AIMessage(content="", tool_calls=[{"name": "query_faq", "args": {"keyword": keyword}, "id": "raw"}])
    deps = RetrievalDependencies(model, Embedder(), Store([hit(17)]),
                                 RerankClient(base_url="https://local.invalid", api_key="test"))
    async def retrieve(query):
        return await search_knowledge_ch04(query, strategy="bm25", dependencies=deps)
    async def check(query, hits):
        assert query == raw
        return {"useful": True, "reason": "有依据"}
    def factory(cid, *, raw_question=None):
        return build_tools(lambda: None, cid, retrieve, check, raw_question=raw_question)
    async with deps:
        service = ChatService(Model(selection), history, factory, ToolExecutor)
        events = [event async for event in service.events(ChatRequest(user_id="u", message=raw))]
    assert events[-1] == ("done", None)
    assert len(requests) == 1
    messages = requests[0]["messages"]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert json.loads(messages[-1]["content"]) == {"current_question": raw}
    saved_call = next(row for row in history.rows if isinstance(row, AIMessage) and row.tool_calls)
    assert saved_call.tool_calls[0]["args"] == {"keyword": keyword}
