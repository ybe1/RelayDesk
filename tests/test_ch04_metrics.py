"""Hand-calculated evidence groups catch ranking and denominator mistakes."""

import pytest


def multi_case():
    return {"bucket": "E_multi", "expected_groups": [
        {"group_id": "fee", "allowed_section_paths": ["fee", "fee-alt"],
         "expect_points": ["满99元包邮", "运费8元"]},
        {"group_id": "return", "allowed_section_paths": ["return"],
         "expect_points": ["七日内退货"]},
    ]}


def test_group_recall_mrr_use_original_rank_despite_context_order_and_tied_scores():
    from app.evaluation.ch04_metrics import retrieval_metrics
    hits = [
        {"rank": 1, "score": 1, "section_path": "other", "answer": "无关"},
        {"rank": 3, "score": 1, "section_path": "fee", "answer": "满 99 元包邮"},
        {"rank": 5, "score": 1, "section_path": "other", "answer": "无关"},
        {"rank": 6, "score": 1, "section_path": "return", "answer": "七日内退货"},
        {"rank": 4, "score": 1, "section_path": "other", "answer": "无关"},
        {"rank": 2, "score": 1, "section_path": "fee-alt", "answer": "满99元包邮"},
    ]
    metrics = retrieval_metrics(multi_case(), hits)
    assert metrics["recall_at_5"] == 0.5
    assert metrics["mrr"] == pytest.approx(1 / 3)  # (1/2 + 1/6) / 2
    assert metrics["evidence_coverage"] == 0.75  # (1/2 + 1) / 2


def test_absent_evidence_is_zero_and_wrong_section_cannot_supply_model_facts():
    from app.evaluation.ch04_metrics import retrieval_metrics
    assert retrieval_metrics(multi_case(), []) == {
        "recall_at_5": 0.0, "mrr": 0.0, "evidence_coverage": 0.0,
    }
    hits = [{"rank": 1, "section_path": "wrong-model", "answer": "满99元包邮 运费8元 七日内退货"}]
    assert retrieval_metrics(multi_case(), hits)["evidence_coverage"] == 0


@pytest.mark.parametrize("hits", [
    [{"section_path": "fee", "answer": ""}],
    [{"rank": 1}, {"rank": 1}], [{"rank": True}], [{"rank": 2}],
])
def test_missing_or_ambiguous_ranks_fail_instead_of_guessing_by_score(hits):
    from app.evaluation.ch04_metrics import retrieval_metrics
    with pytest.raises(ValueError, match="rank"):
        retrieval_metrics(multi_case(), hits)


def test_answer_point_coverage_averages_groups_and_normalizes_only_whitespace():
    from app.evaluation.ch04_metrics import answer_point_coverage
    assert answer_point_coverage(multi_case(), "满 99 元包邮，七日内退货。") == 0.75
    assert answer_point_coverage(multi_case(), "满199元包邮，七日内退货。") == 0.5


def test_d_has_no_retrieval_or_answer_metrics():
    from app.evaluation.ch04_metrics import answer_point_coverage, retrieval_metrics
    case = {"bucket": "D_out_of_scope", "expected_groups": []}
    assert retrieval_metrics(case, []) is None
    assert answer_point_coverage(case, "不知道") is None


def test_refusal_rate_uses_only_completed_answers_and_exposes_empty_denominator():
    from app.evaluation.ch04_metrics import refusal_rate
    assert refusal_rate([{"status": "ok", "refused": True},
                         {"status": "ok", "refused": False},
                         {"status": "skipped", "reason": "TimeoutError"}]) == 0.5
    assert refusal_rate([{"status": "skipped"}]) is None
