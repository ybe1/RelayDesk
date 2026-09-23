from app.knowledge.dedup import qa_fingerprint

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db.models import KnowledgeChunk, QAExtractionStaging


def test_fingerprint_uses_answer_and_normalization():
    assert qa_fingerprint("运费？", " 满 99 包邮。") == qa_fingerprint("运费", "满99包邮")
    assert qa_fingerprint("运费", "满99包邮") != qa_fingerprint("运费", "满199包邮")
    assert qa_fingerprint("邮费多少", "满99包邮") != qa_fingerprint("运费", "满99包邮")


def _session_factory():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as connection:
        connection.exec_driver_sql("""
            CREATE TABLE knowledge_chunks (
                id INTEGER PRIMARY KEY, category VARCHAR(255) NOT NULL,
                questions TEXT NOT NULL, answer TEXT NOT NULL, section_path VARCHAR(512),
                content_type VARCHAR(32), is_key_clause INTEGER NOT NULL DEFAULT 0,
                prev_chunk_id INTEGER, next_chunk_id INTEGER, vector_id VARCHAR(64),
                vectorize_status VARCHAR NOT NULL DEFAULT 'pending',
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.exec_driver_sql("""
            CREATE TABLE qa_extraction_staging (
                id INTEGER PRIMARY KEY, batch_no VARCHAR(64) NOT NULL,
                source_ref VARCHAR(255), question TEXT NOT NULL, answer TEXT NOT NULL,
                status VARCHAR NOT NULL DEFAULT 'extracted',
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
    return sessionmaker(bind=engine, expire_on_commit=False)


def test_deduplicate_staging_keeps_first_exact_pair_across_batches_and_discards_later_copy():
    from app.knowledge.dedup import deduplicate_staging

    factory = _session_factory()
    with factory.begin() as session:
        session.add_all([
            QAExtractionStaging(batch_no="one", source_ref="conversation/1", question="运费？", answer="满 99 元包邮。"),
            QAExtractionStaging(batch_no="two", source_ref="conversation/2", question="运费", answer="满99元包邮"),
        ])

    assert deduplicate_staging(factory) == {"kept": 1, "discarded": 1, "imported": 1}
    with factory() as session:
        rows = session.scalars(select(QAExtractionStaging).order_by(QAExtractionStaging.id)).all()
        chunks = session.scalars(select(KnowledgeChunk)).all()
    assert [row.status for row in rows] == ["kept", "discarded"]
    assert [(chunk.category, chunk.questions, chunk.answer, chunk.section_path, chunk.content_type, chunk.vectorize_status) for chunk in chunks] == [
        ("历史对话", "运费？", "满 99 元包邮。", "conversation/1", "extracted_qa", "pending")
    ]


def test_deduplicate_staging_retains_same_question_with_a_conflicting_answer():
    from app.knowledge.dedup import deduplicate_staging

    factory = _session_factory()
    with factory.begin() as session:
        session.add_all([
            QAExtractionStaging(batch_no="one", source_ref="conversation/1", question="运费？", answer="满 99 元包邮。"),
            QAExtractionStaging(batch_no="two", source_ref="conversation/2", question="运费？", answer="满 199 元包邮。"),
        ])

    assert deduplicate_staging(factory) == {"kept": 2, "discarded": 0, "imported": 2}
    with factory() as session:
        assert [row.status for row in session.scalars(select(QAExtractionStaging).order_by(QAExtractionStaging.id))] == ["kept", "kept"]
        assert [chunk.answer for chunk in session.scalars(select(KnowledgeChunk).order_by(KnowledgeChunk.id))] == ["满 99 元包邮。", "满 199 元包邮。"]


def test_deduplicate_staging_discards_pair_already_in_knowledge():
    from app.knowledge.dedup import deduplicate_staging

    factory = _session_factory()
    with factory.begin() as session:
        session.add(KnowledgeChunk(category="配送", questions="运费", answer="满99元包邮"))
        session.add(QAExtractionStaging(batch_no="one", source_ref="conversation/1", question="运费？", answer="满 99 元包邮。"))

    assert deduplicate_staging(factory) == {"kept": 0, "discarded": 1, "imported": 0}
    with factory() as session:
        assert session.scalar(select(QAExtractionStaging.status)) == "discarded"
        assert len(session.scalars(select(KnowledgeChunk)).all()) == 1
