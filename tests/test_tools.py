import json
import asyncio
import time
from types import SimpleNamespace

from app.tools.business import build_tools


def test_mock_queries_return_requested_identity_and_mock_marker():
    tools = {tool.name: tool for tool in build_tools(lambda: None, 42)}
    for name, args, key, value in (
        ("query_order", {"order_id": "1001"}, "order_id", "1001"),
        ("query_product", {"product_name": "猫粮"}, "product_name", "猫粮"),
        ("query_logistics", {"order_id": "1001"}, "order_id", "1001"),
    ):
        result = json.loads(tools[name].invoke(args))
        assert result["mock"] is True
        assert result[key] == value


def test_order_and_logistics_share_stable_consistent_mock_snapshot():
    snapshots = []
    for _ in range(2):
        tools = {t.name:t for t in build_tools(lambda:None,42)}
        rows = []
        for order_id in map(str, range(1000,1030)):
            logistics = json.loads(tools['query_logistics'].invoke({'order_id':order_id}))
            order = json.loads(tools['query_order'].invoke({'order_id':order_id}))
            if order['status'] == '待发货':
                assert logistics['status'] == '暂无物流'
                assert logistics['location'] is None
            elif order['status'] == '已完成':
                assert logistics['status'] == '已签收'
            else:
                assert order['status'] == '运输中'
                assert logistics['status'] in ['已揽收','运输中','派送中']
            rows.append((order,logistics))
        snapshots.append(rows)
    assert snapshots[0] == snapshots[1]
    assert {row[0]['status'] for row in snapshots[0]} == {'待发货','运输中','已完成'}


def test_ticket_schema_excludes_model_supplied_conversation_id():
    tools = {tool.name: tool for tool in build_tools(lambda: None, 42)}
    schema = tools["create_ticket"].args_schema.model_json_schema()
    assert "conversation_id" not in schema["properties"]
    assert set(tools) == {"query_order", "query_product", "query_logistics", "query_faq", "create_ticket"}


def test_faq_contract_has_no_database_read():
    def forbidden_session():
        raise AssertionError("online retrieval opened MySQL")

    retriever = lambda query: [{
        "id": 12, "score": 0.83, "question": "运费怎么算", "answer": "满99元包邮", "category": "配送",
    }]
    query = {item.name: item for item in build_tools(forbidden_session, 1, retriever=retriever)}["query_faq"]

    result = json.loads(asyncio.run(query.ainvoke({"keyword": "邮费是多少"})))
    assert {key: result[key] for key in ("keyword", "matches")} == {
        "keyword": "邮费是多少",
        "matches": [{"question": "运费怎么算", "answer": "满99元包邮", "category": "配送"}],
    }


def test_faq_preserves_categories_and_empty_semantic_results():
    tools = {item.name: item for item in build_tools(lambda: None, 42, retriever=lambda query: [
        {"id": 3, "score": 0.83, "question": "猫粮保存", "answer": "阴凉处保存", "category": "商品"},
        {"id": 4, "score": 0.72, "question": "退货条件", "answer": "七天内申请", "category": "售后"},
    ])}

    matches = json.loads(asyncio.run(tools["query_faq"].ainvoke({"keyword": "商品和售后"})))["matches"]
    assert matches == [
        {"question": "猫粮保存", "answer": "阴凉处保存", "category": "商品"},
        {"question": "退货条件", "answer": "七天内申请", "category": "售后"},
    ]


def test_ticket_uses_server_conversation_and_commits(monkeypatch):
    from app.repositories import tickets

    calls = []

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def commit(self):
            calls.append("commit")

        def rollback(self):
            calls.append("rollback")

    def save(_session, conversation_id, description, ticket_type):
        calls.append((conversation_id, description, ticket_type))
        return SimpleNamespace(ticket_no="T0001")

    monkeypatch.setattr(tickets, "create_ticket", save)
    ticket = {item.name: item for item in build_tools(Session, 42)}["create_ticket"]
    result = json.loads(ticket.invoke({"description": "请人工处理", "ticket_type": "售后"}))
    assert result["ticket_no"] == "T0001"
    assert calls == [(42, "请人工处理", "售后"), "commit"]

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session
from app.db.models import LowConfidenceQuestion
from app.knowledge.retrieval import RetrievalResult
from app.knowledge.query_understanding import QueryUnderstanding
from app.knowledge.evidence_check import EvidenceDecision
from app.services.tool_executor import ToolExecutor
from tests import ch04_sqlite  # noqa: F401


def retrieval(hits=None, sufficient=True, reason="retrieval_ok"):
    return RetrievalResult(QueryUnderstanding(normalized_query="退货", search_query="退货"),
                           hits if hits is not None else [{"id": 17, "question": "退货条件", "answer": "签收七天内申请", "section_path": "售后/退货", "category": "售后"}],
                           sufficient, reason)


@pytest.fixture
def faq_database():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE conversations (id INTEGER PRIMARY KEY, user_id TEXT)"))
        conn.execute(text("INSERT INTO conversations VALUES (42, 'u')"))
    LowConfidenceQuestion.__table__.create(engine)
    yield engine
    engine.dispose()


@pytest.mark.parametrize("async_retriever", [False, True])
async def test_native_faq_has_exact_evidence_citations_without_mysql(async_retriever):
    def forbidden():
        raise AssertionError("sufficient retrieval must not open MySQL")
    async def retrieve(query):
        return retrieval()
    async def check(query, hits):
        assert query == "原始退货问题"
        assert hits[0]["id"] == 17
        return EvidenceDecision(useful=True, reason="涵盖问题")
    tool = build_tools(forbidden, 42, retrieve if async_retriever else lambda _: retrieval(), check)[3]
    result = json.loads(await tool.ainvoke({"keyword": "原始退货问题"}))
    assert result["sufficient"] is True
    assert result["citations"] == [{"n": 1, "chunk_id": 17, "section_path": "售后/退货", "question": "退货条件", "answer_excerpt": "签收七天内申请"}]
    assert result["evidence"][0]["n"] == 1
    assert result["evidence"][0]["chunk_id"] == 17
    assert result["evidence"][0]["answer"] == "签收七天内申请"


@pytest.mark.parametrize("mechanical", [True, False])
async def test_refusal_inserts_one_row_after_network_and_model_gates(faq_database, mechanical):
    events = []
    def factory():
        events.append("session")
        return Session(faq_database)
    async def retrieve(query):
        events.append("retrieval")
        return retrieval(sufficient=not mechanical, reason="rerank_below_threshold")
    async def check(query, hits):
        events.append("check")
        return EvidenceDecision(useful=False, reason="证据未覆盖退款时效")
    tool = build_tools(factory, 42, retrieve, check)[3]
    raw = "  退款到底几天？  "
    result = json.loads(await tool.ainvoke({"keyword": raw}))
    assert result["sufficient"] is False
    assert result["citations"] == []
    assert result["evidence"] == []
    assert events == (["retrieval", "session"] if mechanical else ["retrieval", "check", "session"])
    with Session(faq_database) as session:
        rows = list(session.scalars(select(LowConfidenceQuestion)))
        assert len(rows) == 1
        assert rows[0].raw_question == raw
        assert rows[0].conversation_id == 42
        assert rows[0].source == ("retrieval_low_conf" if mechanical else "self_check")
        assert rows[0].reason == result["reason"]


async def test_refusal_commit_failure_rolls_back_without_retrying():
    events = []
    class FailingSession:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def add(self, row): events.append("insert")
        def flush(self): pass
        def commit(self):
            events.append("commit")
            raise ConnectionError("uncertain commit")
        def rollback(self): events.append("rollback")
    tool = build_tools(FailingSession, 42, lambda _: retrieval([], False, "no_hits"))[3]
    message = await ToolExecutor([tool], backoff_seconds=0).execute({"name": "query_faq", "args": {"keyword": "没有"}, "id": "c"})
    assert json.loads(message.content)["error"]["code"] == "execution_failed"
    assert events == ["insert", "commit", "rollback"]


async def test_model_failure_is_operational_and_does_not_write_low_confidence():
    def forbidden(): raise AssertionError("opened database on operation failure")
    async def checker(*_): raise ValueError("invalid structured response")
    tool = build_tools(forbidden, 42, lambda _: retrieval(), checker)[3]
    message = await ToolExecutor([tool]).execute({"name": "query_faq", "args": {"keyword": "退货"}, "id": "c"})
    assert json.loads(message.content)["error"]["code"] == "execution_failed"


async def test_async_legacy_retriever_preserves_matches_without_inventing_ids():
    async def retrieve(_): return [{"question": "邮费", "answer": "满99元包邮", "category": "配送"}]
    result = json.loads(await build_tools(lambda: None, 42, retrieve)[3].ainvoke({"keyword": "邮费"}))
    assert result["sufficient"] is True
    assert result["matches"][0]["answer"] == "满99元包邮"
    assert result["citations"] == []


async def test_complete_tool_message_fits_utf8_budget_and_keeps_citation_mapping():
    hits = [{"id": n + 1, "question": '问"\\\n😀' * 1000, "answer": '答"\\\n😀' * 15000, "section_path": "路径" * 500} for n in range(10)]
    async def check(*_): return EvidenceDecision(useful=True, reason="支持")
    tool = build_tools(lambda: None, 42, lambda _: retrieval(hits), check)[3]
    message = await ToolExecutor([tool]).execute({"name": "query_faq", "args": {"keyword": "问题"}, "id": "c"})
    assert len(message.content.encode("utf-8")) <= 60000
    result = json.loads(message.content)["result"]
    assert len(result["citations"]) == 10
    for n, (citation, evidence) in enumerate(zip(result["citations"], result["evidence"]), 1):
        assert set(citation) == {"n", "chunk_id", "section_path", "question", "answer_excerpt"}
        assert citation["n"] == evidence["n"] == n
        assert citation["chunk_id"] == evidence["chunk_id"] == n
        assert citation["answer_excerpt"]
        assert hits[n - 1]["answer"].startswith(citation["answer_excerpt"])
        assert "\ufffd" not in citation["answer_excerpt"]

async def test_refusal_saves_user_verbatim_instead_of_model_keyword(faq_database):
    raw = "  太烦了！我上周退的，钱还没来？\n"
    tool = build_tools(lambda: Session(faq_database), 42, lambda _: retrieval([], False, "no_hits"), raw_question=raw)[3]
    await tool.ainvoke({"keyword": "退款到账"})
    with Session(faq_database) as session:
        assert session.scalar(select(LowConfidenceQuestion)).raw_question == raw


async def test_empty_legacy_result_is_recorded_as_retrieval_refusal(faq_database):
    tool = build_tools(lambda: Session(faq_database), 42, lambda _: [])[3]
    result = json.loads(await tool.ainvoke({"keyword": "天文"}))
    assert result["matches"] == []
    assert result["sufficient"] is False
    with Session(faq_database) as session:
        assert session.scalar(select(LowConfidenceQuestion)).source == "retrieval_low_conf"


async def test_final_evidence_is_exactly_the_evidence_approved_by_checker():
    seen = []
    hits = [{"id": n, "question": "问", "answer": "答" * 1250, "section_path": "路径"} for n in range(10)]
    async def check(_, rows):
        seen.extend(row["answer"] for row in rows)
        return EvidenceDecision(useful=True, reason='"' * 2048)
    tool = build_tools(lambda: None, 42, lambda _: retrieval(hits), check)[3]
    result = json.loads(await tool.ainvoke({"keyword": "问"}))
    assert [row["answer"] for row in result["evidence"]] == seen


async def test_evidence_checker_checks_full_user_question_not_shortened_keyword(faq_database):
    raw = "退货条件是什么？还能保证明天到账吗？"
    async def check(query, hits):
        return EvidenceDecision(useful=query != raw, reason="未覆盖到账保证")
    tool = build_tools(lambda: Session(faq_database), 42, lambda _: retrieval(), check, raw_question=raw)[3]
    result = json.loads(await tool.ainvoke({"keyword": "退货条件"}))
    assert result["sufficient"] is False
    with Session(faq_database) as session:
        assert session.scalar(select(LowConfidenceQuestion)).source == "self_check"


async def test_faq_timeout_after_slow_commit_never_repeats_insert(faq_database):
    from langchain_core.callbacks import AsyncCallbackHandler

    commits = []

    class FinishCallback(AsyncCallbackHandler):
        async def on_tool_end(self, output, **kwargs):
            # Model a yielding tracing callback after the tool body completes.
            await asyncio.sleep(0.01)

    class SlowCommitSession(Session):
        def commit(self):
            super().commit()
            commits.append("committed")
            # The DB has committed while the tool invocation has not returned yet.
            time.sleep(0.08)

    async def retrieve(_):
        return retrieval([], False, "no_hits")

    tool = build_tools(lambda: SlowCommitSession(faq_database), 42, retrieve)[3]
    tool.callbacks = [FinishCallback()]
    result = await ToolExecutor([tool], timeout_seconds=0.03, backoff_seconds=0).execute(
        {"name": "query_faq", "args": {"keyword": "unknown policy"}, "id": "slow-commit"},
    )
    assert json.loads(result.content)["error"]["code"] == "timeout"
    with Session(faq_database) as session:
        rows = list(session.scalars(select(LowConfidenceQuestion)))
        assert len(rows) == 1
        assert rows[0].raw_question == "unknown policy"
    assert commits == ["committed"]


@pytest.mark.parametrize("failure", [TimeoutError, ConnectionError, OSError])
async def test_faq_with_sufficient_retrieval_never_retries_transient_checker_failure(failure):
    calls = []

    def forbidden_session():
        raise AssertionError("operational failure must not write")

    async def retrieve(_):
        calls.append("retrieval")
        return retrieval()

    async def checker(*_):
        calls.append("checker")
        raise failure("transient")

    tool = build_tools(forbidden_session, 42, retrieve, checker)[3]
    result = await ToolExecutor([tool], backoff_seconds=0).execute(
        {"name": "query_faq", "args": {"keyword": "policy"}, "id": "no-write"},
    )
    assert json.loads(result.content)["error"]["code"] == (
        "timeout" if failure is TimeoutError else "execution_failed"
    )
    assert calls == ["retrieval", "checker"]


@pytest.mark.parametrize("failure", [TimeoutError, ConnectionError, OSError])
async def test_sufficient_faq_completion_failure_never_retries(failure):
    from langchain_core.callbacks import AsyncCallbackHandler

    calls = []

    class FailingCompletion(AsyncCallbackHandler):
        raise_error = True

        async def on_tool_end(self, output, **kwargs):
            assert json.loads(output)["sufficient"] is True
            raise failure("completion failed")

    def forbidden_session():
        raise AssertionError("sufficient answer must not write")

    async def retrieve(_):
        calls.append("retrieval")
        return retrieval()

    async def checker(*_):
        return EvidenceDecision(useful=True, reason="covered")

    tool = build_tools(forbidden_session, 42, retrieve, checker)[3]
    tool.callbacks = [FailingCompletion()]
    result = await ToolExecutor([tool], backoff_seconds=0).execute(
        {"name": "query_faq", "args": {"keyword": "policy"}, "id": "completed"},
    )
    assert json.loads(result.content)["error"]["code"] == (
        "timeout" if failure is TimeoutError else "execution_failed"
    )
    assert calls == ["retrieval"]
