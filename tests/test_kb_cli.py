from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db.models import KnowledgeChunk, QAExtractionStaging


def test_ch04_rebuild_holds_writer_lock_and_prints_parity(monkeypatch, capsys):
    from scripts import kb
    events = []
    @contextmanager
    def lock(_):
        events.append('locked')
        yield
        events.append('released')
    monkeypatch.setattr(kb, 'HybridMilvusStore', lambda **_: object(), raising=False)
    monkeypatch.setattr(kb, 'EmbeddingClient', lambda **_: object())
    monkeypatch.setattr(kb, 'settings', SimpleNamespace(milvus_uri='test', embed_base_url='test', embed_api_key='test', embed_model='test'))
    def rebuild(*args):
        events.append('rebuild')
        return {'indexed': 2, 'sql_count': 2, 'milvus_count': 2, 'ids_match': True, 'failed': 0}
    monkeypatch.setattr(kb, 'rebuild_ch04', rebuild, raising=False)
    assert kb.main(['ch04-rebuild'], lock=lock, database_engine=object(), session_factory=object()) == 0
    assert events == ['locked', 'rebuild', 'released']
    assert json.loads(capsys.readouterr().out)['ids_match'] is True


def test_preview_is_read_only_and_reports_all_counters(monkeypatch, capsys):
    from scripts import kb

    @contextmanager
    def forbidden_lock(_engine):
        raise AssertionError("preview must not acquire writer lock")
        yield

    monkeypatch.setattr(kb, "preview", lambda **_: {"imported": 0, "skipped": 0, "indexed": 0, "pending": 2, "failed": 0})

    assert kb.main(["preview"], lock=forbidden_lock) == 0
    assert '"pending": 2' in capsys.readouterr().out


def test_vectorize_holds_writer_lock_and_exposes_crash_hook_only_for_cli(monkeypatch):
    from scripts import kb

    events = []

    @contextmanager
    def lock(_engine):
        events.append("locked")
        yield
        events.append("released")

    class Store:
        def ensure_collection(self):
            events.append("collection")

    monkeypatch.setattr(kb, "MilvusStore", lambda **_: Store())
    monkeypatch.setattr(kb, "EmbeddingClient", lambda **_: object())
    monkeypatch.setattr(kb, "settings", SimpleNamespace(
        milvus_uri="http://milvus.test", embed_base_url="http://embed.test",
        embed_api_key="test", embed_model="BAAI/bge-m3",
    ))
    monkeypatch.setattr(kb, "index_pending", lambda *args, **kwargs: {"indexed": 1, "pending": 0, "failed": 0})

    fake_session_factory = lambda: None
    assert kb.main(["vectorize"], lock=lock, database_engine=object(), session_factory=fake_session_factory) == 0
    assert events == ["locked", "collection", "released"]

    exit_codes = []
    def crashable_index(*args, **kwargs):
        kwargs["after_upsert"]()
        return {"indexed": 1, "pending": 0, "failed": 0}

    monkeypatch.setattr(kb, "index_pending", crashable_index)
    monkeypatch.setattr(kb.os, "_exit", lambda code: exit_codes.append(code))
    assert kb.main(["vectorize", "--crash-after-upsert"], lock=lock, database_engine=object(), session_factory=fake_session_factory) == 0
    assert exit_codes == [86]


def test_vectorize_failure_reports_the_pending_rows_that_remain(monkeypatch, capsys):
    from scripts import kb

    database = create_engine("sqlite+pysqlite:///:memory:")
    with database.begin() as connection:
        connection.exec_driver_sql("""
            CREATE TABLE knowledge_chunks (
                id INTEGER PRIMARY KEY,
                category VARCHAR(255) NOT NULL,
                questions TEXT NOT NULL,
                answer TEXT NOT NULL,
                section_path VARCHAR(512), content_type VARCHAR(32),
                is_key_clause INTEGER NOT NULL DEFAULT 0,
                prev_chunk_id INTEGER, next_chunk_id INTEGER, vector_id VARCHAR(64),
                vectorize_status VARCHAR NOT NULL DEFAULT 'pending',
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
    session_factory = sessionmaker(bind=database, expire_on_commit=False)
    with session_factory.begin() as session:
        session.add_all([
            KnowledgeChunk(category="配送", questions="运费？", answer="满 99 元包邮。"),
            KnowledgeChunk(category="售后", questions="能退吗？", answer="七天内可退。"),
        ])

    class Store:
        def ensure_collection(self):
            pass

    @contextmanager
    def lock(_engine):
        yield

    monkeypatch.setattr(kb, "MilvusStore", lambda **_: Store())
    monkeypatch.setattr(kb, "EmbeddingClient", lambda **_: object())
    monkeypatch.setattr(kb, "settings", SimpleNamespace(
        milvus_uri="http://milvus.test", embed_base_url="http://embed.test",
        embed_api_key="test", embed_model="BAAI/bge-m3",
    ))
    monkeypatch.setattr(kb, "index_pending", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("embed unavailable")))

    assert kb.main(["vectorize"], lock=lock, database_engine=object(), session_factory=session_factory) == 1
    assert json.loads(capsys.readouterr().out)["pending"] == 2


def test_mine_holds_writer_lock_and_runs_extraction_before_global_deduplication(monkeypatch):
    from scripts import kb

    events = []

    @contextmanager
    def lock(_engine):
        events.append("locked")
        yield
        events.append("released")

    monkeypatch.setattr(kb, "get_chat_model", lambda *, disable_thinking: events.append(("model", disable_thinking)) or object())
    monkeypatch.setattr(kb, "extract_batches", lambda factory, model: events.append(("extract", factory, model)) or {
        "batches": 2, "extracted": 3, "skipped": 1,
    })
    monkeypatch.setattr(kb, "deduplicate_staging", lambda factory: events.append(("dedup", factory)) or {
        "kept": 2, "discarded": 1, "imported": 2,
    })
    monkeypatch.setattr(kb, "_pending_count", lambda factory: 2)

    factory = object()
    assert kb.main(["mine"], lock=lock, database_engine=object(), session_factory=factory) == 0
    assert [event if isinstance(event, str) else event[0] for event in events] == [
        "locked", "model", "extract", "dedup", "released"
    ]


def test_seed_conv_only_appends_missing_synthetic_conversations(monkeypatch):
    from scripts import kb
    database = create_engine("sqlite+pysqlite:///:memory:")
    with database.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE conversations (id INTEGER PRIMARY KEY, user_id VARCHAR(64) NOT NULL, status VARCHAR NOT NULL DEFAULT '进行中', created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)")
        connection.exec_driver_sql("CREATE TABLE messages (id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL, role VARCHAR NOT NULL, content TEXT, tool_calls JSON, tool_call_id VARCHAR(64), message_key VARCHAR(128) UNIQUE, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)")
    factory = sessionmaker(bind=database, expire_on_commit=False)
    first = kb.seed_conv(session_factory=factory)
    second = kb.seed_conv(session_factory=factory)
    assert first["imported"] == len(kb.SYNTHETIC_CONVERSATIONS)
    assert second["imported"] == 0


@pytest.mark.parametrize("collection", ["knowledge", "knowledge_ch04"])
def test_reset_requires_confirmation_and_never_touches_ch02_rows(monkeypatch, capsys, collection):
    from scripts import kb
    database = create_engine("sqlite+pysqlite:///:memory:")
    with database.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE knowledge_chunks (id INTEGER PRIMARY KEY, category VARCHAR NOT NULL, questions TEXT NOT NULL, answer TEXT NOT NULL, section_path VARCHAR, content_type VARCHAR, is_key_clause INTEGER NOT NULL DEFAULT 0, prev_chunk_id INTEGER, next_chunk_id INTEGER, vector_id VARCHAR, vectorize_status VARCHAR NOT NULL DEFAULT 'pending', created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)")
        connection.exec_driver_sql("CREATE TABLE qa_extraction_staging (id INTEGER PRIMARY KEY, batch_no VARCHAR NOT NULL, source_ref VARCHAR, question TEXT NOT NULL, answer TEXT NOT NULL, status VARCHAR NOT NULL DEFAULT 'extracted', created_at DATETIME DEFAULT CURRENT_TIMESTAMP)")
        connection.exec_driver_sql("CREATE TABLE faq (id INTEGER PRIMARY KEY, question VARCHAR NOT NULL, answer TEXT NOT NULL, category VARCHAR NOT NULL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)")
    factory = sessionmaker(bind=database, expire_on_commit=False)
    with factory.begin() as session:
        session.add(KnowledgeChunk(category="x", questions="q", answer="a"))
        session.add(QAExtractionStaging(batch_no="b", question="q", answer="a"))
        from app.db.models import FAQ
        session.add(FAQ(id=77, category="ch02", question="旧 FAQ", answer="必须保留"))
    collections = {"knowledge": [1], "knowledge_ch04": [1]}
    class Store:
        def __init__(self, **_): self.cleared = None
        def clear_collection(self, name): collections[name].clear()
    monkeypatch.setattr(kb, "MilvusStore", Store)
    monkeypatch.setattr(kb, "settings", SimpleNamespace(milvus_uri="test", milvus_collection=collection))
    assert kb.main(["reset"], session_factory=factory, database_engine=object()) == 1
    code = kb.main(["reset", "--confirmed"], session_factory=factory, database_engine=object(), lock=lambda _: __import__('contextlib').nullcontext())
    assert code == (1 if collection == "knowledge_ch04" else 0)
    with factory() as session:
        assert len(session.scalars(select(KnowledgeChunk)).all()) == (1 if collection == "knowledge_ch04" else 0)
        assert len(session.scalars(select(QAExtractionStaging)).all()) == (1 if collection == "knowledge_ch04" else 0)
        assert session.get(FAQ, 77).answer == "必须保留"
    if collection == "knowledge_ch04":
        assert "kb-reset is disabled" in capsys.readouterr().err
        assert collections == {"knowledge": [1], "knowledge_ch04": [1]}
        with pytest.raises(ValueError, match="kb-reset is disabled"):
            kb.reset_knowledge(factory, Store())
    else:
        assert collections == {"knowledge": [], "knowledge_ch04": [1]}
