import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import KnowledgeChunk
from app.flywheel.review import CATEGORY, SOURCE_PREFIX
from app.knowledge.embedding import VECTOR_DIMENSION
from app.knowledge.incremental_indexer import index_pending_review_chunks


class Embedder:
    def embed(self, texts):
        self.texts = texts
        return [[0.0] * VECTOR_DIMENSION for _ in texts]


class Store:
    def __init__(self): self.calls = []
    def upsert(self, rows): self.calls.append(rows)


def _sessions():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with engine.begin() as connection:
        connection.exec_driver_sql("""CREATE TABLE knowledge_chunks (
          id INTEGER PRIMARY KEY AUTOINCREMENT, category VARCHAR(255) NOT NULL, questions TEXT NOT NULL,
          answer TEXT NOT NULL, section_path VARCHAR(512), content_type VARCHAR(32), is_key_clause INTEGER DEFAULT 0,
          prev_chunk_id INTEGER, next_chunk_id INTEGER, vector_id VARCHAR(64), vectorize_status VARCHAR(64) DEFAULT 'pending',
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    return sessionmaker(engine, expire_on_commit=False)


def _add(factory, *, review=True):
    with factory() as session:
        row = KnowledgeChunk(category=CATEGORY if review else "旧数据", questions="邮费是多少",
            answer="满99元包邮", section_path=(f"{SOURCE_PREFIX}4" if review else "旧文档 / 1"),
            content_type="faq", is_key_clause=0, vectorize_status="pending")
        session.add(row); session.commit(); return row.id


def test_recovery_only_indexes_review_sources_with_hybrid_contract():
    factory = _sessions(); approved = _add(factory); legacy = _add(factory, review=False)
    embedder, store = Embedder(), Store()
    report = index_pending_review_chunks(factory, embedder, store)
    assert list(report.indexed_ids) == [approved]
    row = store.calls[0][0]
    assert set(row) == {"id", "dense", "text", "question", "answer", "category",
                        "section_path", "content_type", "is_key_clause"}
    assert "sparse" not in row
    assert row["text"] == "分类：用户问题补充\n问题：邮费是多少\n答案：满99元包邮"
    with factory() as session:
        assert session.get(KnowledgeChunk, approved).vectorize_status == "done"
        assert session.get(KnowledgeChunk, approved).vector_id == str(approved)
        assert session.get(KnowledgeChunk, legacy).vectorize_status == "pending"


def test_remote_upsert_then_sql_crash_is_safe_to_retry():
    factory = _sessions(); approved = _add(factory); store = Store()
    with pytest.raises(RuntimeError):
        index_pending_review_chunks(factory, Embedder(), store,
            after_upsert=lambda: (_ for _ in ()).throw(RuntimeError("crash")))
    with factory() as session:
        assert session.get(KnowledgeChunk, approved).vectorize_status == "pending"
    report = index_pending_review_chunks(factory, Embedder(), store)
    assert list(report.indexed_ids) == [approved]
    assert [call[0]["id"] for call in store.calls] == [approved, approved]


def test_invalid_limit_and_oversize_utf8_fail_before_remote_write():
    factory = _sessions(); approved = _add(factory); store = Store()
    with factory() as session:
        session.get(KnowledgeChunk, approved).answer = "猫" * 20_000
        session.commit()
    with pytest.raises(ValueError, match="answer exceeds"):
        index_pending_review_chunks(factory, Embedder(), store)
    assert store.calls == []
    with pytest.raises(ValueError, match="limit"):
        index_pending_review_chunks(factory, Embedder(), store, limit=0)
