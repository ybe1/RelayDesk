import math

import pytest


def _config(threshold=0.26):
    from app.flywheel.contracts import ConfidenceConfig

    return ConfidenceConfig(
        threshold=threshold, top1_floor=0.0, top1_ceiling=1.0,
        gap_floor=0.0, gap_ceiling=0.5, effective_score_threshold=0.5,
        top_k=5, calibration_id="cal-v1",
    )


def _hits():
    return [
        {"id": 1, "rank": 1, "rerank_score": 0.9, "section_path": "returns/window", "is_key_clause": True},
        {"id": 2, "rank": 2, "rerank_score": 0.7, "section_path": "returns/cost", "is_key_clause": False},
        {"id": 3, "rank": 3, "rerank_score": 0.6, "section_path": "returns/cost", "is_key_clause": False},
    ]


def test_confidence_uses_fixed_weights_and_distinct_sections():
    from app.flywheel.confidence import WEIGHTS, compute_confidence

    decision = compute_confidence(_hits(), _config())
    assert WEIGHTS == {"top1": 0.5, "evidence_count": 0.2, "gap": 0.2, "key_clause": 0.1}
    assert decision.signals.top1 == pytest.approx(0.9)
    assert decision.signals.evidence_count == pytest.approx(2 / 5)
    assert decision.signals.gap == pytest.approx(0.4)
    assert decision.signals.key_clause == 1.0
    assert decision.score == pytest.approx(0.5 * 0.9 + 0.2 * (2 / 5) + 0.2 * 0.4 + 0.1)
    assert decision.allowed is True


def test_no_hits_is_a_valid_low_confidence_decision():
    from app.flywheel.confidence import compute_confidence

    decision = compute_confidence([], _config())
    assert decision.allowed is False
    assert decision.score == 0
    assert decision.reason == "no_hits"


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, "0.9", None])
def test_invalid_rerank_score_is_operational_error(bad):
    from app.flywheel.confidence import InvalidConfidenceInput, compute_confidence

    hits = _hits()
    hits[0]["rerank_score"] = bad
    with pytest.raises(InvalidConfidenceInput, match="rerank_score"):
        compute_confidence(hits, _config())


def test_single_hit_uses_frozen_missing_gap_value_zero():
    from app.flywheel.confidence import compute_confidence

    decision = compute_confidence([_hits()[0]], _config())
    assert decision.signals.gap == 0.0


def test_config_rejects_invalid_normalization_ranges():
    from app.flywheel.contracts import ConfidenceConfig

    with pytest.raises(ValueError):
        ConfidenceConfig(
            threshold=0.26, top1_floor=1, top1_ceiling=1,
            gap_floor=0, gap_ceiling=1, effective_score_threshold=0.5,
            top_k=5, calibration_id="bad",
        )
