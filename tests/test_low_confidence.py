"""Transaction contract for the low-confidence question pool."""

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from app.db.models import LowConfidenceQuestion
from app.repositories.low_confidence import insert_low_confidence
from tests import ch04_sqlite  # noqa: F401 - register MySQL type adapters for SQLite DDL


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE conversations (id INTEGER PRIMARY KEY, user_id TEXT NOT NULL)"))
    LowConfidenceQuestion.__table__.create(engine)
    with Session(engine) as db:
        yield db
    engine.dispose()


@pytest.mark.parametrize("source", ["retrieval_low_conf", "self_check", "user_feedback"])
def test_insert_preserves_exact_question_and_nullable_conversation(session, source):
    raw = "  呃，退货？\n别改我的原话！  "
    row = insert_low_confidence(session, None, raw, source, "证据不足")
    assert row.id is not None
    assert row.conversation_id is None
    assert row.raw_question == raw
    assert row.source == source
    assert row.reason == "证据不足"
    assert session.scalar(select(LowConfidenceQuestion).where(LowConfidenceQuestion.id == row.id)) is row
    assert row.created_at is not None


def test_insert_joins_callers_transaction_and_rollback(session):
    row = insert_low_confidence(session, None, "原话", "self_check", None)
    row_id = row.id
    session.rollback()
    assert session.get(LowConfidenceQuestion, row_id) is None


def test_insert_can_link_existing_conversation(session):
    session.execute(text("INSERT INTO conversations (id, user_id) VALUES (23, 'test')"))
    row = insert_low_confidence(session, 23, "怎么回事", "user_feedback", None)
    assert row.conversation_id == 23


def test_invalid_source_is_rejected_before_write(session):
    with pytest.raises(ValueError, match="source"):
        insert_low_confidence(session, None, "原话", "other", None)
    assert session.scalar(select(LowConfidenceQuestion)) is None
