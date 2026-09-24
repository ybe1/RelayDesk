"""Faith-case ledger persistence and status transitions."""

from datetime import datetime

import pytest
from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.orm import Session

from app.db.models import FaithCase
from app.repositories.faith_cases import (
    Conflict,
    faith_case_status_map,
    list_faith_cases,
    set_faith_case_status,
    upsert_faith_case,
)
from tests import ch04_sqlite  # noqa: F401 - register MySQL type adapters for SQLite DDL


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    FaithCase.__table__.create(engine)
    with Session(engine) as db:
        yield db
    engine.dispose()


def put(session, eval_id="A43", **changes):
    values = dict(eval_id=eval_id, bucket="A_policy", query="能退货吗", strategy="hybrid_rerank",
                  answer="错误答案", reason="引用没有支持结论", citations=[{"n": 1, "chunk_id": 7}],
                  judge_model="judge-v1")
    values.update(changes)
    return upsert_faith_case(session, **values)


def test_upsert_one_row_per_eval_replaces_all_snapshots_and_increments(session):
    first = put(session)
    first_seen = first.first_seen_at
    original_id = first.id
    first.last_seen_at = datetime(2020, 1, 1)
    session.flush()
    second = put(session, bucket="B_model", query="新问题", strategy="bm25", answer="新答案",
                 reason="新理由", citations=[{"n": 2, "chunk_id": 9}], judge_model="judge-v2")
    assert second.id == original_id
    assert session.scalar(select(func.count()).select_from(FaithCase)) == 1
    assert second.seen_count == 2
    assert (second.bucket, second.query, second.strategy, second.answer, second.reason) == (
        "B_model", "新问题", "bm25", "新答案", "新理由")
    assert second.citations == [{"n": 2, "chunk_id": 9}]
    assert second.judge_model == "judge-v2"
    assert second.first_seen_at == first_seen
    assert second.last_seen_at > datetime(2020, 1, 1)


def test_upsert_nullable_citations_replace_prior_snapshot(session):
    put(session)
    row = put(session, citations=None, judge_model=None)
    assert row.citations is None
    assert row.judge_model is None


def test_resolved_recurrence_reopens_but_retains_resolution_history(session):
    row = put(session)
    set_faith_case_status(row.id, "已解决", "补了政策条款", expected_seen_count=row.seen_count, session=session)
    resolved_at = row.resolved_at
    assert resolved_at is not None
    reopened = put(session, answer="再次编造")
    assert reopened.status == "未解决"
    assert reopened.resolution is None
    assert reopened.resolved_at == resolved_at
    assert reopened.seen_count == 2


@pytest.mark.parametrize("status", ["已解决", "无需解决"])
def test_terminal_status_needs_meaningful_resolution(session, status):
    row = put(session)
    for resolution in (None, "", " \n "):
        with pytest.raises(ValueError, match="resolution"):
            set_faith_case_status(row.id, status, resolution, expected_seen_count=row.seen_count, session=session)
    assert row.status == "未解决"
    assert row.resolved_at is None


def test_resolution_is_trimmed_and_limited_to_300_characters(session):
    row = put(session)
    set_faith_case_status(row.id, "已解决", "  " + "a" * 300 + "  ", expected_seen_count=row.seen_count, session=session)
    assert row.resolution == "a" * 300
    with pytest.raises(ValueError, match="resolution"):
        set_faith_case_status(row.id, "已解决", "  " + "b" * 301 + "  ", expected_seen_count=row.seen_count, session=session)
    assert row.resolution == "a" * 300


def test_stale_admin_status_update_cannot_overwrite_eval_recurrence(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'faith.db'}")
    FaithCase.__table__.create(engine)
    try:
        with Session(engine) as setup:
            case_id = put(setup).id
            setup.commit()
        with Session(engine) as admin_read:
            version_read_by_admin = admin_read.get(FaithCase, case_id).seen_count
        with Session(engine) as evaluator:
            put(evaluator, answer="新的编造答案")
            evaluator.commit()
        with Session(engine) as admin_write:
            with pytest.raises(Conflict):
                set_faith_case_status(
                    case_id, "已解决", "旧页面的处置", expected_seen_count=version_read_by_admin,
                    session=admin_write,
                )
            row = admin_write.get(FaithCase, case_id)
            assert row.status == "未解决"
            assert row.resolution is None
            assert row.answer == "新的编造答案"
            assert row.seen_count == 2
    finally:
        engine.dispose()


def test_deleted_case_in_identity_map_reports_missing_not_conflict(session):
    row = put(session)
    case_id = row.id
    version = row.seen_count
    session.execute(
        delete(FaithCase).where(FaithCase.id == case_id)
        .execution_options(synchronize_session=False)
    )
    assert row in session
    with pytest.raises(LookupError, match="not found"):
        set_faith_case_status(
            case_id, "已解决", "旧页面处置", expected_seen_count=version, session=session,
        )


def test_status_change_clears_resolution_when_reopened_and_rejects_unknown(session):
    row = put(session)
    set_faith_case_status(row.id, "无需解决", "裁判误判", expected_seen_count=row.seen_count, session=session)
    resolved_at = row.resolved_at
    assert row.resolution == "裁判误判"
    set_faith_case_status(row.id, "未解决", "忽略此文本", expected_seen_count=row.seen_count, session=session)
    assert row.resolution is None
    assert row.resolved_at == resolved_at
    with pytest.raises(ValueError, match="status"):
        set_faith_case_status(row.id, "unknown", "原因", expected_seen_count=row.seen_count, session=session)
    with pytest.raises(LookupError):
        set_faith_case_status(99999, "未解决", None, expected_seen_count=1, session=session)


def test_upsert_and_status_changes_rollback_with_caller(session):
    row = put(session)
    set_faith_case_status(row.id, "已解决", "已修复", expected_seen_count=row.seen_count, session=session)
    session.rollback()
    assert session.scalar(select(FaithCase)) is None


def test_list_sorts_latest_first_then_id_and_paginates(session):
    a = put(session, "A1")
    b = put(session, "A2")
    c = put(session, "A3")
    a.last_seen_at = datetime(2024, 1, 1)
    b.last_seen_at = datetime(2025, 1, 1)
    c.last_seen_at = datetime(2025, 1, 1)
    session.flush()
    rows, total = list_faith_cases(None, 1, 2, session=session)
    assert total == 3
    assert [r.eval_id for r in rows] == ["A3", "A2"]
    rows, total = list_faith_cases(None, 2, 2, session=session)
    assert total == 3
    assert [r.eval_id for r in rows] == ["A1"]
    set_faith_case_status(a.id, "已解决", "补充了证据", expected_seen_count=a.seen_count, session=session)
    rows, total = list_faith_cases("已解决", 1, 10, session=session)
    assert total == 1
    assert [r.eval_id for r in rows] == ["A1"]


def test_status_map_returns_only_requested_known_eval_ids(session):
    a = put(session, "A1")
    put(session, "A2")
    set_faith_case_status(a.id, "无需解决", "误判", expected_seen_count=a.seen_count, session=session)
    assert faith_case_status_map(["A1", "missing"], session=session) == {"A1": "无需解决"}
    assert faith_case_status_map([], session=session) == {}


@pytest.mark.parametrize("status,page,size", [("bad", 1, 10), (None, 0, 10), (None, 1, 0)])
def test_list_rejects_invalid_filters_and_pages(session, status, page, size):
    with pytest.raises(ValueError):
        list_faith_cases(status, page, size, session=session)
