import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import FaithCase, KnowledgeChunk
from app.main import app
from tests import ch04_sqlite  # noqa: F401
from tests.test_faith_cases_repository import put


@pytest.fixture
def api(monkeypatch):
    from app.api import rageval, kb
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    FaithCase.__table__.create(engine)
    with engine.begin() as connection:
        connection.exec_driver_sql("""CREATE TABLE knowledge_chunks (
          id INTEGER PRIMARY KEY, category TEXT, questions TEXT, answer TEXT, section_path TEXT,
          content_type TEXT, is_key_clause INTEGER DEFAULT 0, prev_chunk_id INTEGER, next_chunk_id INTEGER,
          vector_id TEXT, vectorize_status TEXT DEFAULT 'pending',
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    factory = sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(rageval, "get_session_factory", lambda: factory)
    monkeypatch.setattr(kb, "get_session_factory", lambda: factory)
    monkeypatch.setattr(kb, "get_store", lambda: pytest.fail("chunk details cannot use Milvus"))
    yield TestClient(app), factory
    engine.dispose()


def test_pages_filters_and_full_snapshot(api):
    client, factory = api
    snapshot = [{"n": 1, "chunk_id": 7, "answer": "全文" * 900}]
    with factory.begin() as session:
        put(session, "A1", citations=snapshot)
        row = put(session, "A2")
        row.status, row.resolution = "无需解决", "人工确认有依据"
    result = client.get("/api/rag-eval/faith-cases", params={"page": 2, "size": 1}).json()
    assert (result["total"], result["page"], result["size"]) == (2, 2, 1)
    assert result["items"][0]["citations"] == snapshot
    assert result["items"][0]["seen_count"] == 1
    assert result["items"][0]["recurrence"] is False
    assert client.get("/api/rag-eval/faith-cases", params={"status": "无需解决"}).json()["total"] == 1


@pytest.mark.parametrize("params", [{"status": "bad"}, {"page": 0}, {"size": 0}, {"size": 101}])
def test_bad_filters_and_page_bounds(api, params):
    assert api[0].get("/api/rag-eval/faith-cases", params=params).status_code == 422


@pytest.mark.parametrize("payload", [{"status": "bad", "expected_seen_count": 1},
    {"status": "已解决", "expected_seen_count": 1},
    {"status": "无需解决", "resolution": "  ", "expected_seen_count": 1},
    {"status": "已解决", "resolution": "x" * 301, "expected_seen_count": 1},
    {"status": "未解决"}, {"status": "未解决", "expected_seen_count": True}])
def test_status_validation(api, payload):
    assert api[0].post("/api/rag-eval/faith-cases/1/status", json=payload).status_code == 422


def test_status_commits_recurrence_and_stale_version_is_conflict(api):
    client, factory = api
    with factory.begin() as session:
        case_id = put(session).id
    url = f"/api/rag-eval/faith-cases/{case_id}/status"
    response = client.post(url, json={"status": "已解决", "resolution": " 已补条款 ", "expected_seen_count": 1})
    assert response.status_code == 200
    assert response.json()["resolution"] == "已补条款"
    with factory.begin() as session:
        assert session.get(FaithCase, case_id).status == "已解决"
        put(session, answer="再次编造")
    conflict = client.post(url, json={"status": "已解决", "resolution": "旧操作", "expected_seen_count": 1})
    assert conflict.status_code == 409
    item = client.get("/api/rag-eval/faith-cases").json()["items"][0]
    assert item["status"] == "未解决" and item["seen_count"] == 2 and item["recurrence"] is True
    assert item["answer"] == "再次编造" and item["resolution"] is None and item["resolved_at"]
    assert client.post("/api/rag-eval/faith-cases/999/status", json={"status": "未解决", "expected_seen_count": 1}).status_code == 404


def test_chunk_detail_returns_mysql_full_text_and_unknown_404(api):
    client, factory = api
    with factory.begin() as session:
        session.add(KnowledgeChunk(id=7, category="条款", questions="多个问题\n第二问", answer="完整原文" * 900,
                                   section_path="手册 / 条款", content_type="policy", is_key_clause=1))
    response = client.get("/api/kb/chunks/7")
    assert response.status_code == 200
    assert response.json()["answer"] == "完整原文" * 900
    assert response.json()["questions"] == "多个问题\n第二问"
    assert response.json()["section_path"] == "手册 / 条款"
    assert client.get("/api/kb/chunks/999").status_code == 404
