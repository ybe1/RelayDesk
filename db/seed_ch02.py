"""Repeatable Ch02 FAQ seed. Run with `python -m db.seed_ch02`."""
from sqlalchemy import select

from app.db.models import FAQ
from app.db.session import SessionLocal


FAQ_ROWS = [
    ("退货政策是什么", "商品签收后 7 天内，如符合退货条件，可联系客服申请退货。具体是否符合条件以人工审核为准。", "售后"),
    ("如何申请退货", "请提供订单号和退货原因，客服会协助发起退货申请。", "售后"),
]


def seed() -> int:
    inserted = 0
    with SessionLocal.begin() as session:
        for question, answer, category in FAQ_ROWS:
            if session.scalar(select(FAQ.id).where(FAQ.question == question)) is None:
                session.add(FAQ(question=question, answer=answer, category=category))
                inserted += 1
    return inserted


if __name__ == "__main__":
    print(f"Inserted {seed()} FAQ rows")
