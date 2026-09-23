"""Recoverable write path from the SQL source of truth to Milvus."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import func, select

from app.db.models import KnowledgeChunk


@dataclass(frozen=True)
class _PendingRow:
    id: int
    category: str
    questions: str
    answer: str


def _pending_count(session_factory: Callable) -> int:
    with session_factory() as session:
        return int(session.scalar(
            select(func.count()).select_from(KnowledgeChunk).where(
                KnowledgeChunk.vectorize_status == "pending"
            )
        ) or 0)


def _next_pending_batch(session_factory: Callable, batch_size: int) -> list[_PendingRow]:
    """Read and detach one batch before making any remote call."""
    with session_factory() as session:
        rows = session.scalars(
            select(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == "pending")
            .order_by(KnowledgeChunk.id)
            .limit(batch_size)
        ).all()
        return [_PendingRow(row.id, row.category, row.questions, row.answer) for row in rows]


def _mark_done(session_factory: Callable, ids: list[int]) -> None:
    with session_factory.begin() as session:
        rows = session.scalars(
            select(KnowledgeChunk).where(
                KnowledgeChunk.id.in_(ids), KnowledgeChunk.vectorize_status == "pending"
            )
        ).all()
        for row in rows:
            row.vectorize_status = "done"
            row.vector_id = str(row.id)


def index_pending(
    session_factory: Callable,
    embedder,
    store,
    batch_size: int = 16,
    after_upsert=None,
) -> dict[str, int]:
    """Upsert pending chunks by their SQL ID, then record completed batches.

    A remote failure or process crash before ``_mark_done`` deliberately leaves
    SQL rows pending. A later run repeats an idempotent Milvus upsert with the
    same IDs and can safely finish the SQL acknowledgement.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    indexed = 0
    while rows := _next_pending_batch(session_factory, batch_size):
        texts = [row.category + "\n" + row.questions + "\n" + row.answer for row in rows]
        vectors = embedder.embed(texts)
        store.upsert([
            {
                "id": row.id,
                "vector": vector,
                "question": row.questions,
                "answer": row.answer,
                "category": row.category,
            }
            for row, vector in zip(rows, vectors, strict=True)
        ])
        if after_upsert is not None:
            after_upsert()
        _mark_done(session_factory, [row.id for row in rows])
        indexed += len(rows)

    return {"indexed": indexed, "pending": _pending_count(session_factory), "failed": 0}
