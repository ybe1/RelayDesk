from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import FAQ


def search_faq(session: Session, keyword: str, limit: int = 5) -> list[FAQ]:
    escaped = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return list(session.scalars(select(FAQ).where(FAQ.question.like(f"%{escaped}%", escape="\\")).order_by(FAQ.id).limit(limit)))
