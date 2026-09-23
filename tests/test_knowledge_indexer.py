import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db.models import KnowledgeChunk


@pytest.fixture
def session_factory():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as connection:
        connection.exec_driver_sql("""
            CREATE TABLE knowledge_chunks (
                id INTEGER PRIMARY KEY,
                category VARCHAR(255) NOT NULL,
                questions TEXT NOT NULL,
                answer TEXT NOT NULL,
                section_path VARCHAR(512),
                content_type VARCHAR(32),
                is_key_clause INTEGER NOT NULL DEFAULT 0,
                prev_chunk_id INTEGER,
                next_chunk_id INTEGER,
                vector_id VARCHAR(64),
                vectorize_status VARCHAR NOT NULL DEFAULT 'pending',
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory.begin() as session:
        session.add_all([
            KnowledgeChunk(category="配送", questions="运费多少？", answer="满 99 元包邮。"),
            KnowledgeChunk(category="售后", questions="如何退货？", answer="七天内可申请。"),
        ])
    return factory


class FakeEmbedder:
    def __init__(self):
        self.texts = []

    def embed(self, texts):
        self.texts.append(texts)
        return [[float(index)] * 1024 for index in range(len(texts))]


class FakeStore:
    def __init__(self):
        self.rows_by_id = {}

    def upsert(self, rows):
        for row in rows:
            self.rows_by_id[row["id"]] = row


def _status_rows(session_factory):
    with session_factory() as session:
        return session.scalars(select(KnowledgeChunk).order_by(KnowledgeChunk.id)).all()


def test_index_pending_recovers_after_crash_that_follows_idempotent_upsert(session_factory):
    from app.knowledge.indexer import index_pending

    embedder = FakeEmbedder()
    store = FakeStore()

    def crash_after_upsert():
        raise RuntimeError("injected-after-upsert")

    with pytest.raises(RuntimeError, match="injected-after-upsert"):
        index_pending(session_factory, embedder, store, after_upsert=crash_after_upsert)

    assert [row.vectorize_status for row in _status_rows(session_factory)] == ["pending", "pending"]
    assert set(store.rows_by_id) == {1, 2}

    result = index_pending(session_factory, embedder, store)

    assert result == {"indexed": 2, "pending": 0, "failed": 0}
    assert [(row.vectorize_status, row.vector_id) for row in _status_rows(session_factory)] == [
        ("done", "1"), ("done", "2")
    ]
    assert set(store.rows_by_id) == {1, 2}
    assert embedder.texts[0] == ["配送\n运费多少？\n满 99 元包邮。", "售后\n如何退货？\n七天内可申请。"]


@pytest.mark.parametrize("failure", ["embed", "upsert"])
def test_index_pending_leaves_rows_pending_when_remote_work_fails(session_factory, failure):
    from app.knowledge.indexer import index_pending

    class BrokenEmbedder(FakeEmbedder):
        def embed(self, texts):
            if failure == "embed":
                raise RuntimeError("embedding unavailable")
            return super().embed(texts)

    class BrokenStore(FakeStore):
        def upsert(self, rows):
            if failure == "upsert":
                raise RuntimeError("vector store unavailable")
            super().upsert(rows)

    with pytest.raises(RuntimeError):
        index_pending(session_factory, BrokenEmbedder(), BrokenStore())

    assert [row.vectorize_status for row in _status_rows(session_factory)] == ["pending", "pending"]
