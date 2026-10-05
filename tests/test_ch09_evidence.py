from datetime import UTC, datetime

import pytest


def _hits():
    return [
        {"id": 4, "rank": 4, "score": 0.14, "rerank_score": 0.41, "section_path": "p/4", "question": "q4", "answer": "a4", "is_key_clause": False},
        {"id": 2, "rank": 2, "score": 0.32, "rerank_score": 0.88, "section_path": "p/2", "question": "q2", "answer": "a2", "is_key_clause": True},
        {"id": 1, "rank": 1, "score": 0.36, "rerank_score": 0.93, "section_path": "p/1", "question": "q1", "answer": "a1", "is_key_clause": False},
        {"id": 3, "rank": 3, "score": 0.21, "rerank_score": 0.66, "section_path": "p/3", "question": "q3", "answer": "a3", "is_key_clause": False},
    ]


def test_snapshot_sorts_by_retrieval_rank_and_keeps_full_answer():
    from app.flywheel.evidence import snapshot_hits

    snapshot = snapshot_hits(
        "猫窝能机洗吗", "hybrid_rerank", _hits(), retrieval_id="ret-1",
        created_at=datetime(2026, 10, 5, tzinfo=UTC), query_sources=["猫窝清洗", "猫窝机洗"],
    )
    assert [hit.rank for hit in snapshot.hits] == [1, 2, 3, 4]
    assert snapshot.hits[0].answer == "a1"
    assert snapshot.query_sources == ["猫窝清洗", "猫窝机洗"]


def test_pool_snapshot_has_exact_three_state_contract():
    from app.flywheel.contracts import EvidenceSnapshot
    from app.flywheel.evidence import pool_snapshot, snapshot_hits

    assert pool_snapshot(None, retrieval_completed=False) is None
    empty = EvidenceSnapshot(
        retrieval_id="ret-empty", query="q", strategy="hybrid_rerank",
        created_at=datetime(2026, 10, 5, tzinfo=UTC), hits=[], query_sources=[],
    )
    assert pool_snapshot(empty, retrieval_completed=True) == []
    saved = pool_snapshot(snapshot_hits(
        "q", "hybrid_rerank", _hits(), retrieval_id="ret-2",
        created_at=datetime(2026, 10, 5, tzinfo=UTC),
    ), retrieval_completed=True)
    assert [row["rank"] for row in saved] == [1, 2, 3]
    assert all({"chunk_id", "section_path", "question", "answer", "score", "rerank_score", "rank"} <= row.keys() for row in saved)


def test_pool_snapshot_rejects_inconsistent_completion_state():
    from app.flywheel.evidence import pool_snapshot

    with pytest.raises(ValueError, match="completed retrieval requires a snapshot"):
        pool_snapshot(None, retrieval_completed=True)


def test_snapshot_rejects_duplicate_or_non_contiguous_rank():
    from app.flywheel.evidence import snapshot_hits

    hits = _hits()
    hits[1]["rank"] = 1
    with pytest.raises(ValueError, match="rank"):
        snapshot_hits("q", "hybrid_rerank", hits, retrieval_id="ret-3",
                      created_at=datetime(2026, 10, 5, tzinfo=UTC))


def test_snapshot_rejects_non_boolean_key_clause_metadata():
    from app.flywheel.evidence import snapshot_hits

    hits = _hits()
    hits[0]["is_key_clause"] = "false"
    with pytest.raises(ValueError, match="is_key_clause"):
        snapshot_hits("q", "hybrid_rerank", hits, retrieval_id="ret-4",
                      created_at=datetime(2026, 10, 5, tzinfo=UTC))

