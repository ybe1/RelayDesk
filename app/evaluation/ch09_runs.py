from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db.models import EvalRun


class EvalRunResult(BaseModel):
    triggered_by: Literal["定时", "手动"]
    dataset_size: int = Field(gt=0)
    metrics: dict[str, Any]


class EvalRunRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def save_complete(self, result: EvalRunResult) -> EvalRun:
        if result.metrics.get("completed") is not True:
            raise ValueError("only complete evaluation runs can be saved")
        run_id = result.metrics.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("complete evaluation run requires run_id")
        with self.session_factory() as session:
            with session.begin():
                existing = next((row for row in session.scalars(select(EvalRun)).all()
                                 if row.metrics.get("run_id") == run_id), None)
                if existing is not None:
                    return existing
                row = EvalRun(triggered_by=result.triggered_by,
                              dataset_size=result.dataset_size, metrics=result.metrics)
                session.add(row)
                session.flush()
                return row

    def list(self, limit: int = 50) -> list[EvalRun]:
        with self.session_factory() as session:
            return list(session.scalars(select(EvalRun).order_by(EvalRun.created_at.desc(), EvalRun.id.desc()).limit(limit)))


def result_from_ch04_report(report: dict, *, run_id: str, triggered_by: str,
                            scheduled_slot: str | None = None) -> EvalRunResult:
    strategies = {}
    for name, values in report.get("retrieval", {}).items():
        overall = values.get("overall", {})
        strategies[name] = {
            "recall_at_5": overall.get("recall_at_5"),
            "mrr": overall.get("mrr"),
            "completed": overall.get("completed", 0),
            "skipped": overall.get("skipped", 0),
            "buckets": values.get("buckets", {}),
        }
    meta = report.get("meta", {})
    generation = report.get("generation", {})
    faith = generation.get("faithfulness", {})
    refusal = generation.get("strategies", {}).get("hybrid_rerank", {}).get("buckets", {}).get(
        "D_out_of_scope", {}).get("refusal_rate")
    complete = meta.get("retrieval_status") == "complete" and generation.get("status") == "complete"
    configuration = meta.get("configuration", {})
    configuration_hash=hashlib.sha256(json.dumps(configuration,sort_keys=True,ensure_ascii=False,
        separators=(",",":")).encode("utf-8")).hexdigest()
    metrics = {
        "run_id": run_id,
        "dataset_hash": meta.get("dataset_sha256"),
        "corpus_hash": configuration.get("corpus_sha256"),
        "strategies": strategies,
        "recall_at_k": 5,
        "faithfulness": faith.get("faithfulness_rate"),
        "refusal_rate": refusal,
        "completed": complete,
        "failed": meta.get("retrieval_skipped", 0) + faith.get("skipped", 0),
        "denominators": {"dataset": meta.get("case_count"), "faithfulness": faith.get("completed", 0)},
        "scheduled_slot": scheduled_slot,
        "configuration": configuration,
        "configuration_hash": configuration_hash,
    }
    return EvalRunResult(triggered_by=triggered_by, dataset_size=int(meta.get("case_count", 0)), metrics=metrics)


@dataclass
class EvaluationDependencies:
    repository: EvalRunRepository
    report_dir: Path
    evaluator: Any
    scheduled_slot: str | None = None


def run_ch09_evaluation(triggered_by: str, run_id: str, deps: EvaluationDependencies) -> EvalRunResult:
    """Run once, or repair the SQL row from an already-complete report."""
    report_path = Path(deps.report_dir) / f"{run_id.replace(':', '_')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
    complete = report is not None and report.get("meta",{}).get("retrieval_status")=="complete" \
        and report.get("generation",{}).get("status")=="complete"
    if not complete:
        report = deps.evaluator(report_path)
        if report is None: report = json.loads(report_path.read_text(encoding="utf-8"))
    result = result_from_ch04_report(report, run_id=run_id, triggered_by=triggered_by,
                                    scheduled_slot=deps.scheduled_slot)
    deps.repository.save_complete(result)
    return result

