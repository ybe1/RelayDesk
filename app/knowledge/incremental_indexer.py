from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select

from app.db.models import KnowledgeChunk
from app.flywheel.review import CATEGORY, SOURCE_PREFIX
from app.knowledge.hybrid_store import validate_text_fields


@dataclass(frozen=True)
class IndexReport:
    indexed_ids: tuple[int, ...]
    pending: int


def index_pending_review_chunks(session_factory, embedder, store, *, limit: int = 100,
                                after_upsert=None) -> IndexReport:
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    with session_factory() as session:
        rows = list(session.scalars(select(KnowledgeChunk).where(
            KnowledgeChunk.vectorize_status == "pending",
            KnowledgeChunk.category == CATEGORY,
            KnowledgeChunk.section_path.like(f"{SOURCE_PREFIX}%"),
        ).order_by(KnowledgeChunk.id).limit(limit)))
        detached = [{"id": row.id, "category": row.category, "questions": row.questions,
            "answer": row.answer, "section_path": row.section_path or "",
            "content_type": row.content_type or "", "is_key_clause": bool(row.is_key_clause)} for row in rows]
    if not detached:
        return IndexReport((), 0)
    texts = [f"分类：{row['category']}\n问题：{row['questions']}\n答案：{row['answer']}" for row in detached]
    vectors = embedder.embed(texts)
    payload = []
    for row, vector, text_value in zip(detached, vectors, texts, strict=True):
        item = {"id": row["id"], "question": row["questions"], "answer": row["answer"],
            "category": row["category"], "text": text_value,
            "section_path": row["section_path"], "content_type": row["content_type"],
            "is_key_clause": row["is_key_clause"], "dense": vector}
        validate_text_fields(item)
        payload.append(item)
    store.upsert(payload)
    if after_upsert is not None:
        after_upsert()
    ids = [row["id"] for row in detached]
    with session_factory() as session:
        confirmed = list(session.scalars(select(KnowledgeChunk).where(
            KnowledgeChunk.id.in_(ids), KnowledgeChunk.vectorize_status == "pending").with_for_update()))
        for row in confirmed:
            row.vector_id = str(row.id)
            row.vectorize_status = "done"
        session.commit()
        pending = len(ids) - len(confirmed)
    return IndexReport(tuple(ids), pending)
