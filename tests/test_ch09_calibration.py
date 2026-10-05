import json
from pathlib import Path

import pytest


DATASET = Path("data/ch09/confidence_cases.jsonl")


def test_frozen_manifest_has_60_answerable_and_20_refusals():
    assert DATASET.is_file()
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 80
    assert sum(row["evidence_sufficient"] is True for row in rows) == 60
    assert sum(row["evidence_sufficient"] is False for row in rows) == 20
    assert all(row["fact_group"] and row["reason"] for row in rows)
    assert {row["bucket"] for row in rows if row["evidence_sufficient"]} == {
        "A_policy", "B_model", "C_colloquial", "E_multi",
    }


def test_selector_prefers_zero_false_pass_then_lowest_false_block():
    from app.evaluation.confidence_calibration import ScoredCase, select_threshold

    rows = [
        *[ScoredCase(case_id=f"yes-{i}", fact_group=f"g-{i}", expected_allowed=True, score=score)
          for i, score in enumerate([0.90, 0.80, 0.70, 0.60])],
        ScoredCase(case_id="no-1", fact_group="outside-1", expected_allowed=False, score=0.25),
        ScoredCase(case_id="no-2", fact_group="outside-2", expected_allowed=False, score=0.10),
    ]
    report = select_threshold(rows, candidate_thresholds=[0.20, 0.26, 0.65, 0.95], metadata={
        "dataset_hash": "d", "corpus_hash": "c", "rerank_model": "r", "top_k": 5,
    })
    assert report.threshold == 0.26
    assert report.false_pass_rate == 0
    assert report.false_block_count == 0


def test_selector_rejects_all_refuse_solution():
    from app.evaluation.confidence_calibration import NoSafeUsableThreshold, ScoredCase, select_threshold

    rows = [
        ScoredCase(case_id="yes", fact_group="inside", expected_allowed=True, score=0.1),
        ScoredCase(case_id="no", fact_group="outside", expected_allowed=False, score=0.9),
    ]
    with pytest.raises(NoSafeUsableThreshold):
        select_threshold(rows, candidate_thresholds=[0.95], metadata={
            "dataset_hash": "d", "corpus_hash": "c", "rerank_model": "r", "top_k": 5,
        })


def test_report_contains_confusion_and_both_f_scores():
    from app.evaluation.confidence_calibration import ScoredCase, select_threshold

    rows = [
        ScoredCase(case_id="yes", fact_group="inside", expected_allowed=True, score=0.9),
        ScoredCase(case_id="no", fact_group="outside", expected_allowed=False, score=0.1),
    ]
    report = select_threshold(rows, candidate_thresholds=[0.5], metadata={
        "dataset_hash": "d", "corpus_hash": "c", "rerank_model": "r", "top_k": 5,
    })
    assert report.confusion == {"tp": 1, "fp": 0, "tn": 1, "fn": 0}
    assert report.f1 == 1
    assert report.f_0_5 == 1


def test_report_compatibility_rejects_model_topk_or_corpus_change(tmp_path):
    from app.evaluation.confidence_calibration import (
        ConfidenceConfigurationError, ScoredCase, load_confidence_config,
        save_report, select_threshold,
    )

    report = select_threshold([
        ScoredCase(case_id="yes", fact_group="inside", expected_allowed=True, score=0.9),
        ScoredCase(case_id="no", fact_group="outside", expected_allowed=False, score=0.1),
    ], candidate_thresholds=[0.5], metadata={
        "dataset_hash": "d", "corpus_hash": "c", "rerank_model": "r", "top_k": 5,
    })
    path = tmp_path / "report.json"
    save_report(report, path)
    assert load_confidence_config(path, corpus_hash="c", rerank_model="r", top_k=5).threshold == 0.5
    with pytest.raises(ConfidenceConfigurationError):
        load_confidence_config(path, corpus_hash="changed", rerank_model="r", top_k=5)
    with pytest.raises(ConfidenceConfigurationError):
        load_confidence_config(path, corpus_hash="c", rerank_model="changed", top_k=5)
    with pytest.raises(ConfidenceConfigurationError):
        load_confidence_config(path, corpus_hash="c", rerank_model="r", top_k=10)
