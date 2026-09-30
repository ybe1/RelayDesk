"""Read saved evaluation reports and administer the faith-case ledger."""

import json
import math
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.repositories.faith_cases import Conflict, list_faith_cases, set_faith_case_status

router = APIRouter()
REPORT_PATH = Path(__file__).resolve().parents[2] / "data/ch04/reports/rag_eval.json"


def get_session_factory():
    from app.db.session import SessionLocal
    return SessionLocal


Status = Literal["未解决", "已解决", "无需解决"]


class StatusRequest(BaseModel):
    status: Status
    resolution: str | None = None
    expected_seen_count: int = Field(ge=1, strict=True)


def case_payload(row):
    return {**{column.name: getattr(row, column.name) for column in row.__table__.columns},
            "recurrence": row.status == "未解决" and row.resolved_at is not None}


def _reject_nonfinite(value):
    raise ValueError("non-finite number in report")


def _validate_finite_values(value):
    """JSON exponent overflow can yield infinity without parse_constant."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite number in report")
    if isinstance(value, dict):
        for child in value.values():
            _validate_finite_values(child)
    elif isinstance(value, list):
        for child in value:
            _validate_finite_values(child)


def _saved_best(report):
    best, retrieval = report.get("best"), report.get("retrieval")
    if not isinstance(best, dict) or not isinstance(retrieval, dict):
        return None
    strategy, mrr = best.get("strategy"), best.get("mrr")
    if strategy not in ("vector", "bm25", "hybrid", "hybrid_rerank"):
        return None
    values = retrieval.get(strategy)
    overall = values.get("overall") if isinstance(values, dict) else None
    if not isinstance(overall, dict):
        return None
    completed, count = overall.get("completed"), overall.get("case_count")
    if (type(completed) is not int or type(count) is not int or count <= 0 or completed != count
            or type(mrr) not in (int, float) or not math.isfinite(mrr) or not 0 <= mrr <= 1
            or type(overall.get("mrr")) not in (int, float) or overall["mrr"] != mrr):
        return None
    return {"strategy": strategy, "mrr": mrr}


@router.get("/api/rag-eval/overview")
def overview():
    unavailable = {"report_status": "missing", "report": None, "best_mrr": None}
    try:
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
        if not isinstance(report, dict):
            raise ValueError("report must be an object")
        _validate_finite_values(report)
    except FileNotFoundError:
        return unavailable
    except (OSError, ValueError):
        return {**unavailable, "report_status": "corrupt"}
    # Only expose the persisted winner after validating its complete denominator.
    best = _saved_best(report)
    generation = report.get("generation", {})
    meta = report.get("meta", {})
    complete = (isinstance(generation, dict) and generation.get("status") == "complete"
                and isinstance(meta, dict) and meta.get("retrieval_status") == "complete")
    return {"report_status": "complete" if complete else "partial", "report": report, "best_mrr": best}


@router.get("/api/rag-eval/faith-cases")
def faith_cases(status: Status | None = None, page: int = Query(1, ge=1), size: int = Query(20, ge=1, le=100)):
    with get_session_factory()() as session:
        rows, total = list_faith_cases(status, page, size, session=session)
        return {"items": [case_payload(row) for row in rows], "total": total, "page": page, "size": size}


@router.post("/api/rag-eval/faith-cases/{case_id}/status")
def change_status(case_id: int, request: StatusRequest):
    try:
        with get_session_factory().begin() as session:
            row = set_faith_case_status(case_id, request.status, request.resolution,
                expected_seen_count=request.expected_seen_count, session=session)
            result = case_payload(row)
        return result
    except Conflict as error:
        raise HTTPException(409, "faith case changed; reload before reviewing") from error
    except LookupError as error:
        raise HTTPException(404, "faith case not found") from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
