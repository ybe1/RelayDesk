"""Repeatable full Ch04 indexing without changing Ch03 acknowledgements."""

from sqlalchemy import select

from app.db.models import KnowledgeChunk
from app.knowledge.hybrid_store import validate_text_fields


def rebuild_ch04(session_factory, embedder, store, batch_size: int = 16, after_upsert=None) -> dict:
    """Caller holds writer_lock; SQL IDs make every completed remote batch repeatable."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    store.ensure_collection()
    indexed, last_id, sql_ids = 0, None, set()
    while True:
        with session_factory() as session:
            query = select(KnowledgeChunk).order_by(KnowledgeChunk.id).limit(batch_size)
            if last_id is not None:
                query = query.where(KnowledgeChunk.id > last_id)
            source = session.scalars(query).all()
            rows = [{"id": r.id, "question": r.questions, "answer": r.answer, "category": r.category,
                     "text": f"分类：{r.category}\n问题：{r.questions}\n答案：{r.answer}",
                     "section_path": r.section_path or "", "content_type": r.content_type or "",
                     "is_key_clause": bool(r.is_key_clause)} for r in source]
        if not rows:
            break
        for row in rows:
            validate_text_fields(row)
        vectors = embedder.embed([r["text"] for r in rows])
        store.upsert([{**row, "dense": vector} for row, vector in zip(rows, vectors, strict=True)])
        if after_upsert is not None:
            after_upsert()
        sql_ids.update(row["id"] for row in rows)
        last_id = rows[-1]["id"]
        indexed += len(rows)
    count, remote_ids = store.count(), store.ids()
    if count != len(sql_ids) or remote_ids != sql_ids:
        raise RuntimeError(f"Ch04 parity failed: SQL={len(sql_ids)}, Milvus={count}, "
                           f"missing={sorted(sql_ids - remote_ids)}, extra={sorted(remote_ids - sql_ids)}")
    return {"indexed": indexed, "sql_count": len(sql_ids), "milvus_count": count, "ids_match": True, "failed": 0}
