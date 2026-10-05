"""Persistence for refusal and feedback questions; the caller owns the transaction."""

from sqlalchemy.orm import Session

from app.db.models import LowConfidenceQuestion


SOURCES = frozenset({"retrieval_low_conf", "self_check", "user_feedback"})


def insert_low_confidence(
    session: Session, conversation_id: int | None, raw_question: str,
    source: str, reason: str | None,
    *, retrieved_chunks: list[dict] | dict | None = None,
) -> LowConfidenceQuestion:
    if source not in SOURCES:
        raise ValueError("invalid low-confidence source")
    row = LowConfidenceQuestion(
        conversation_id=conversation_id, raw_question=raw_question,
        source=source, reason=reason, retrieved_chunks=retrieved_chunks,
    )
    session.add(row)
    session.flush()
    return row
