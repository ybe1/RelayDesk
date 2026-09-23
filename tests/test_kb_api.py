from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from app.db.models import KnowledgeChunk, QAExtractionStaging
from app.main import app


@pytest.fixture
def session_factory():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    @event.listens_for(engine, "connect")
    def foreign_keys(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")
    with engine.begin() as connection:
        connection.exec_driver_sql("""CREATE TABLE knowledge_chunks (
          id INTEGER PRIMARY KEY, category VARCHAR(255) NOT NULL, questions TEXT NOT NULL,
          answer TEXT NOT NULL, section_path VARCHAR(512), content_type VARCHAR(32),
          is_key_clause INTEGER NOT NULL DEFAULT 0, prev_chunk_id INTEGER,
          next_chunk_id INTEGER, vector_id VARCHAR(64), vectorize_status VARCHAR NOT NULL DEFAULT 'pending',
          created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
        connection.exec_driver_sql("""CREATE TABLE qa_extraction_staging (
          id INTEGER PRIMARY KEY, batch_no VARCHAR(64) NOT NULL, source_ref VARCHAR(255),
          question TEXT NOT NULL, answer TEXT NOT NULL, status VARCHAR NOT NULL DEFAULT 'extracted',
          created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
    return sessionmaker(bind=engine, expire_on_commit=False)


def _client(monkeypatch, session_factory):
    from app.api import kb
    monkeypatch.setattr(kb, "get_session_factory", lambda: session_factory)
    monkeypatch.setattr(kb, "get_database_engine", lambda: object())
    return TestClient(app)


def test_preview_is_read_only_and_rejects_invalid_manual_input(monkeypatch, session_factory):
    client = _client(monkeypatch, session_factory)
    before = session_factory().scalars(select(KnowledgeChunk)).all()
    response = client.post("/api/kb/preview", json={
        "markdown": "# 配送\n满 99 元包邮。", "content_type": "policy", "source_name": "manual.md"
    })
    assert response.status_code == 200
    assert response.json()["chunks"][0]["questions"] == "配送"
    assert session_factory().scalars(select(KnowledgeChunk)).all() == before
    assert client.post("/api/kb/preview", json={"markdown": "", "content_type": "policy", "source_name": "x.md"}).status_code == 400
    assert client.post("/api/kb/preview", json={"markdown": "# x\n正文", "content_type": "bad", "source_name": "x.md"}).status_code == 400
    assert client.post("/api/kb/preview", json={"markdown": "# x\n正文", "content_type": "policy", "source_name": "../x.md"}).status_code == 400


def test_ingest_locks_writer_and_is_idempotent(monkeypatch, session_factory):
    from app.api import kb
    events = []
    @contextmanager
    def lock(_engine):
        events.append("locked")
        yield
        events.append("released")
    monkeypatch.setattr(kb, "writer_lock", lock)
    client = _client(monkeypatch, session_factory)
    payload = {"markdown": "# 配送\n同标题的第一段。\n\n|项目|规则|\n|-|-|\n|运费|8元|", "content_type": "policy", "source_name": "manual.md"}
    first = client.post("/api/kb/ingest", json=payload)
    second = client.post("/api/kb/ingest", json=payload)
    assert first.json() == {"inserted": 2, "skipped": 0, "indexed": 0, "pending": 2}
    assert second.json()["inserted"] == 0
    assert events == ["locked", "released", "locked", "released"]


def test_ingest_keeps_committed_pending_rows_when_vectorization_fails(monkeypatch, session_factory):
    from app.api import kb
    @contextmanager
    def lock(_engine):
        yield
    monkeypatch.setattr(kb, "writer_lock", lock)
    monkeypatch.setattr(kb, "index_pending", lambda *args: (_ for _ in ()).throw(RuntimeError("embed down")))
    response = _client(monkeypatch, session_factory).post("/api/kb/ingest", json={
        "markdown": "# 售后\n七天内可以退货。", "content_type": "manual", "source_name": "manual.md", "vectorize": True
    })
    assert response.status_code == 502
    assert response.json()["detail"] == {"inserted": 1, "skipped": 0, "indexed": 0, "pending": 1, "message": "原文已入库，向量待补"}
    with session_factory() as session:
        assert session.scalars(select(KnowledgeChunk)).one().vectorize_status == "pending"


def test_overview_returns_unknown_consistency_if_one_dependency_fails(monkeypatch, session_factory):
    from app.api import kb
    monkeypatch.setattr(kb, "get_store", lambda: (_ for _ in ()).throw(RuntimeError("milvus down")))
    response = _client(monkeypatch, session_factory).get("/api/kb/overview")
    assert response.status_code == 200
    assert response.json()["consistent"] is None
    assert response.json()["dependencies"]["milvus"] == "unavailable"


def test_overview_uses_a_read_only_store_and_hides_uri_credentials(monkeypatch, session_factory):
    from app.api import kb
    class Store:
        def ensure_collection(self):
            raise AssertionError("GET overview must not create, load, or mutate collections")
        def count(self): return 0
    monkeypatch.setattr(kb, "get_store", lambda: Store())
    response = _client(monkeypatch, session_factory).get("/api/kb/overview")
    assert response.status_code == 200
    assert kb.safe_endpoint("http://user:secret@milvus.internal:19530/path") == "http://milvus.internal:19530"


def test_staging_and_recent_bound_pagination(monkeypatch, session_factory):
    with session_factory.begin() as session:
        session.add(QAExtractionStaging(batch_no="a", question="q", answer="a", status="extracted"))
        session.add(KnowledgeChunk(category="配送", questions="q", answer="a"))
    client = _client(monkeypatch, session_factory)
    assert client.get("/api/kb/staging?limit=101").status_code == 422
    assert client.get("/api/kb/recent?limit=101").status_code == 422
    assert client.get("/api/kb/staging?status=bad").status_code == 422
