from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db.models import LowConfidenceQuestion
from app.flywheel.pool import GapEvent, record_gap


def test_mysql_advisory_lock_name_fits_server_limit():
    from app.flywheel.pool import gap_lock_name
    assert len(gap_lock_name("turn-1:feedback")) <= 64
    assert gap_lock_name("turn-1:feedback") == gap_lock_name("turn-1:feedback")


def factory(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'pool.db'}")
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE conversations (id INTEGER PRIMARY KEY, user_id VARCHAR(64), status VARCHAR(16), created_at DATETIME, updated_at DATETIME)")
        connection.exec_driver_sql("CREATE TABLE review_queue (id INTEGER PRIMARY KEY)")
        connection.exec_driver_sql("CREATE TABLE low_confidence_questions (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id INTEGER, raw_question TEXT NOT NULL, source VARCHAR(32) NOT NULL, reason TEXT, retrieved_chunks JSON, matched_review_id INTEGER, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)")
    sessions = sessionmaker(engine, expire_on_commit=False)
    with engine.begin() as connection:
        connection.exec_driver_sql("INSERT INTO conversations (id,user_id,status) VALUES (1,'u','进行中')")
    return sessions


def event(key="turn-1:feedback", *, chunks=None):
    return GapEvent(1, "这个回答没解决", "user_feedback", "用户点踩",
                    chunks, key)


def test_same_event_is_idempotent_but_a_new_turn_is_distinct(tmp_path):
    sessions = factory(tmp_path)
    first = record_gap(event(), sessions)
    assert record_gap(event(), sessions) == first
    second = record_gap(event("turn-2:feedback"), sessions)
    assert second != first
    with sessions() as session:
        assert len(list(session.scalars(select(LowConfidenceQuestion)))) == 2


def test_snapshot_preserves_null_empty_and_array(tmp_path):
    sessions = factory(tmp_path)
    ids = [record_gap(event(f"k-{i}", chunks=value), sessions)
           for i, value in enumerate((None, [], [{"chunk_id": 7, "rerank_score": 0.8}]))]
    with sessions() as session:
        rows = [session.get(LowConfidenceQuestion, row_id) for row_id in ids]
        assert [row.retrieved_chunks for row in rows] == [None, [], [{"chunk_id": 7, "rerank_score": 0.8}]]
