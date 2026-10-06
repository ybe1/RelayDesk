from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

import json

from app.evaluation.ch09_runs import (
    EvaluationDependencies, EvalRunRepository, EvalRunResult, run_ch09_evaluation,
)


def repository():
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text("""CREATE TABLE eval_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, triggered_by VARCHAR(8) NOT NULL,
            dataset_size INTEGER NOT NULL, metrics JSON NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL)"""))
    return EvalRunRepository(sessionmaker(bind=engine, expire_on_commit=False))


def result(run_id, *, complete=True):
    return EvalRunResult(triggered_by="手动", dataset_size=300, metrics={
        "run_id": run_id, "completed": complete, "dataset_hash": "dataset",
        "corpus_hash": "corpus", "strategies": {}, "recall_at_k": 5,
        "faithfulness": 0.9, "refusal_rate": 1.0, "failed": 0,
    })


def test_two_runs_persist_and_same_run_id_is_idempotent():
    repo = repository()
    first = repo.save_complete(result("run-1"))
    duplicate = repo.save_complete(result("run-1"))
    second = repo.save_complete(result("run-2"))
    assert first.id == duplicate.id
    assert second.id != first.id
    assert [row.metrics["run_id"] for row in repo.list()] == ["run-2", "run-1"]


def test_partial_run_is_not_saved_and_complete_report_repairs_missing_row():
    repo = repository()
    try:
        repo.save_complete(result("partial", complete=False))
    except ValueError as error:
        assert "complete" in str(error)
    else:
        raise AssertionError("partial result must be rejected")
    assert repo.list() == []
    repaired = repo.save_complete(result("report-exists"))
    assert repaired.metrics["run_id"] == "report-exists"


def test_existing_complete_report_repairs_row_without_rerunning_paid_evaluation(tmp_path):
    repo = repository()
    report = {
        "meta": {"case_count": 1, "dataset_sha256": "d", "retrieval_status": "complete",
                 "retrieval_skipped": 0, "configuration": {"corpus_sha256": "c"}},
        "retrieval": {name: {"overall": {"recall_at_5": 1, "mrr": 1, "completed": 1, "skipped": 0}}
                      for name in ("vector", "bm25", "hybrid", "hybrid_rerank")},
        "generation": {"status": "complete", "faithfulness": {"faithfulness_rate": 1, "completed": 1, "skipped": 0},
                       "strategies": {"hybrid_rerank": {"buckets": {"D_out_of_scope": {"refusal_rate": 1}}}}},
    }
    (tmp_path / "repair.json").write_text(json.dumps(report), encoding="utf-8")
    called = False
    def paid(_path):
        nonlocal called; called = True
    saved = run_ch09_evaluation("手动", "repair", EvaluationDependencies(repo, tmp_path, paid))
    assert saved.metrics["completed"] is True
    assert called is False
    assert repo.list()[0].metrics["run_id"] == "repair"

