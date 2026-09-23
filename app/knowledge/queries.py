"""Read-only ledger queries for the Ch03 management API."""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import func, select

from app.db.models import KnowledgeChunk, QAExtractionStaging


def knowledge_counts(session_factory: Callable) -> dict[str, int]:
    with session_factory() as session:
        rows = dict(session.execute(
            select(KnowledgeChunk.vectorize_status, func.count()).group_by(KnowledgeChunk.vectorize_status)
        ).all())
        return {
            "total": int(sum(rows.values())),
            "pending": int(rows.get("pending", 0)),
            "done": int(rows.get("done", 0)),
            "key_clause_count": int(session.scalar(
                select(func.count()).select_from(KnowledgeChunk).where(KnowledgeChunk.is_key_clause == 1)
            ) or 0),
        }


def staging_counts(session_factory: Callable) -> dict[str, int]:
    with session_factory() as session:
        rows = dict(session.execute(
            select(QAExtractionStaging.status, func.count()).group_by(QAExtractionStaging.status)
        ).all())
    return {state: int(rows.get(state, 0)) for state in ("extracted", "kept", "discarded")}


def recent_chunks(session_factory: Callable, *, limit: int, offset: int) -> list[dict]:
    with session_factory() as session:
        rows = session.scalars(select(KnowledgeChunk).order_by(KnowledgeChunk.id.desc()).limit(limit).offset(offset)).all()
    return [{
        "id": row.id, "category": row.category, "questions": row.questions,
        "answer": row.answer, "section_path": row.section_path,
        "content_type": row.content_type, "vectorize_status": row.vectorize_status,
        "is_key_clause": bool(row.is_key_clause),
    } for row in rows]


def staging_rows(session_factory: Callable, *, status: str | None, limit: int, offset: int) -> list[dict]:
    statement = select(QAExtractionStaging).order_by(QAExtractionStaging.id.desc()).limit(limit).offset(offset)
    if status:
        statement = statement.where(QAExtractionStaging.status == status)
    with session_factory() as session:
        rows = session.scalars(statement).all()
    return [{
        "id": row.id, "batch_no": row.batch_no, "source_ref": row.source_ref,
        "question": row.question, "answer": row.answer, "status": row.status,
    } for row in rows]
