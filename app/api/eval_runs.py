from fastapi import APIRouter, Query, Request

from app.db.session import SessionLocal
from app.evaluation.ch09_runs import EvalRunRepository


router = APIRouter(prefix="/api/eval-runs", tags=["evaluation"])


@router.get("")
def list_eval_runs(request: Request, limit: int = Query(20, ge=1, le=100)):
    repository = getattr(request.app.state, "eval_run_repository", None) or EvalRunRepository(SessionLocal)
    rows = repository.list(limit)
    return {"items": [{"id": row.id, "triggered_by": row.triggered_by,
                       "dataset_size": row.dataset_size, "metrics": row.metrics,
                       "created_at": row.created_at.isoformat()} for row in rows]}
