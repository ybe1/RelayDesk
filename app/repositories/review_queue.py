from sqlalchemy import select

from app.db.models import LowConfidenceQuestion, ReviewQueue


def list_all_candidates(session, *, page_size: int = 200) -> list[ReviewQueue]:
    rows, cursor = [], 0
    while True:
        page = list(session.scalars(select(ReviewQueue).where(ReviewQueue.id > cursor)
            .order_by(ReviewQueue.id).limit(page_size)))
        if not page:
            return rows
        rows.extend(page)
        cursor = page[-1].id


def pending_gaps(session, limit: int) -> list[LowConfidenceQuestion]:
    return list(session.scalars(select(LowConfidenceQuestion)
        .where(LowConfidenceQuestion.matched_review_id.is_(None))
        .order_by(LowConfidenceQuestion.id).limit(limit)))
