import pytest
from types import SimpleNamespace
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from app.db.models import FAQ, KnowledgeChunk
from app.knowledge.chunks import ChunkDraft
from app.knowledge.importer import import_chunks, import_legacy_faq, legacy_faq_draft


@pytest.fixture
def session_factory():
    engine = create_engine("sqlite+pysqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")

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
                prev_chunk_id INTEGER REFERENCES knowledge_chunks(id) ON DELETE SET NULL,
                next_chunk_id INTEGER REFERENCES knowledge_chunks(id) ON DELETE SET NULL,
                vector_id VARCHAR(64),
                vectorize_status VARCHAR NOT NULL DEFAULT 'pending',
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.exec_driver_sql("""
            CREATE TABLE faq (
                id INTEGER PRIMARY KEY,
                question VARCHAR(512) NOT NULL,
                answer TEXT NOT NULL,
                category VARCHAR(64) NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
    return sessionmaker(bind=engine, expire_on_commit=False)


def draft(answer: str, *, title: str = "配送") -> ChunkDraft:
    return ChunkDraft("商品 FAQ", title, answer, "products.md / 商品 FAQ / 配送", "faq")


def test_import_is_idempotent_and_links_only_new_source_chunks(session_factory):
    result = import_chunks(session_factory, [draft("第一段。"), draft("第二段。")])
    duplicate = import_chunks(session_factory, [draft("第一段。"), draft("第二段。")])

    assert result.inserted == 2
    assert result.skipped == 0
    assert duplicate.inserted == 0
    assert duplicate.skipped == 2
    with session_factory() as session:
        chunks = session.scalars(select(KnowledgeChunk).order_by(KnowledgeChunk.id)).all()
    assert chunks[0].next_chunk_id == chunks[1].id
    assert chunks[1].prev_chunk_id == chunks[0].id


def test_import_preserves_same_title_with_different_bodies(session_factory):
    result = import_chunks(session_factory, [draft("条件 A。"), draft("条件 B。")])
    assert result.inserted == 2


def test_import_rolls_back_everything_when_a_draft_is_too_long(session_factory):
    too_long = ChunkDraft("x" * 256, "配送", "正文。", "products.md / 商品 FAQ", "faq")
    with pytest.raises(ValueError, match="category"):
        import_chunks(session_factory, [draft("有效。"), too_long])

    with session_factory() as session:
        assert session.scalars(select(KnowledgeChunk)).all() == []


def test_legacy_faq_is_copied_read_only_into_a_stable_source_draft():
    faq = SimpleNamespace(id=42, category="配送", question="多久送达？", answer="通常两天内送达。")
    draft = legacy_faq_draft(faq)

    assert draft.category == "配送"
    assert draft.questions == "多久送达？"
    assert draft.answer == "通常两天内送达。"
    assert draft.section_path == "legacy-faq/42"
    assert draft.content_type == "faq"


def test_import_legacy_faq_reads_without_mutating_ch02_rows(session_factory):
    with session_factory.begin() as session:
        session.add(FAQ(id=7, category="配送", question="多久送达？", answer="通常两天内送达。"))

    first = import_legacy_faq(session_factory)
    second = import_legacy_faq(session_factory)

    assert first.inserted == 1
    assert second.inserted == 0
    with session_factory() as session:
        faq = session.get(FAQ, 7)
        chunk = session.scalars(select(KnowledgeChunk)).one()
    assert (faq.category, faq.question, faq.answer) == ("配送", "多久送达？", "通常两天内送达。")
    assert chunk.section_path == "legacy-faq/7"
