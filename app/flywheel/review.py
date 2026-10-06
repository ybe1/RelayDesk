from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select, text

from app.db.models import KnowledgeChunk, ReviewQueue
from app.flywheel.locks import FLYWHEEL_LOCK_NAME, LOCAL_FLYWHEEL_LOCK
from app.knowledge.locks import writer_lock
from contextlib import nullcontext


CATEGORY = "用户问题补充"
SOURCE_PREFIX = "数据飞轮 / review_queue/"


class ReviewConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class ReviewResult:
    review_id: int
    review_status: str
    chunk_id: int | None
    vectorize_status: str | None


def approve_review(session_factory, review_id: int, approved_answer: str,
                   expected_status: str = "待审") -> ReviewResult:
    if expected_status != "待审":
        raise ValueError("only pending reviews can be decided")
    answer = approved_answer.strip() if isinstance(approved_answer, str) else ""
    if not answer or len(answer.encode("utf-8")) > 49_152:
        raise ValueError("approved_answer must be nonempty and fit the knowledge answer field")
    section = f"{SOURCE_PREFIX}{review_id}"
    engine=session_factory.kw.get("bind")
    knowledge_lock=writer_lock(engine) if engine.dialect.name == "mysql" else nullcontext()
    # Knowledge lock is always outermost; rebuild and review approval cannot interleave.
    with knowledge_lock:
        return _under_flywheel_lock(session_factory, _approve_transaction,
            review_id, answer, expected_status, section)


def _approve_transaction(session_factory, review_id, answer, expected_status, section):
    with session_factory() as session:
        review = session.scalar(select(ReviewQueue).where(ReviewQueue.id == review_id).with_for_update())
        if review is None:
            raise LookupError("review not found")
        existing = session.scalar(select(KnowledgeChunk).where(KnowledgeChunk.section_path == section).with_for_update())
        if review.review_status == "通过":
            if review.approved_answer != answer or existing is None:
                raise ReviewConflict("approved review answer or knowledge row does not match")
            return ReviewResult(review.id, review.review_status, existing.id, existing.vectorize_status)
        if review.review_status != expected_status:
            raise ReviewConflict("review status changed")
        if existing is not None:
            raise ReviewConflict("knowledge source already exists")
        if len(review.normalized_question) > 512 or len(review.normalized_question.encode("utf-8")) > 8_192:
            raise ValueError("normalized question exceeds knowledge field")
        chunk = KnowledgeChunk(category=CATEGORY, questions=review.normalized_question,
            answer=answer, section_path=section, content_type="faq", is_key_clause=0,
            vectorize_status="pending")
        session.add(chunk); session.flush()
        review.review_status = "通过"
        review.approved_answer = answer
        session.commit()
        return ReviewResult(review.id, review.review_status, chunk.id, chunk.vectorize_status)


def reject_review(session_factory, review_id: int, expected_status: str = "待审") -> ReviewResult:
    if expected_status != "待审":
        raise ValueError("only pending reviews can be decided")
    return _under_flywheel_lock(session_factory, _reject_transaction, review_id, expected_status)


def _reject_transaction(session_factory, review_id, expected_status):
    with session_factory() as session:
        review = session.scalar(select(ReviewQueue).where(ReviewQueue.id == review_id).with_for_update())
        if review is None:
            raise LookupError("review not found")
        if review.review_status == "驳回":
            return ReviewResult(review.id, review.review_status, None, None)
        if review.review_status != expected_status:
            raise ReviewConflict("review status changed")
        review.review_status = "驳回"
        session.commit()
        return ReviewResult(review.id, review.review_status, None, None)


def _under_flywheel_lock(session_factory, operation, *args):
    lock_session = session_factory()
    mysql = lock_session.bind is not None and lock_session.bind.dialect.name == "mysql"
    acquired = False
    try:
        if mysql:
            acquired = lock_session.execute(text("SELECT GET_LOCK(:name, 0)"),
                {"name": FLYWHEEL_LOCK_NAME}).scalar() == 1
        else:
            acquired = LOCAL_FLYWHEEL_LOCK.acquire(blocking=False)
        if not acquired:
            raise ReviewConflict("flywheel review is busy")
        return operation(session_factory, *args)
    finally:
        if mysql and acquired:
            lock_session.execute(text("SELECT RELEASE_LOCK(:name)"), {"name": FLYWHEEL_LOCK_NAME})
        elif acquired:
            LOCAL_FLYWHEEL_LOCK.release()
        lock_session.close()
