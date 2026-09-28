from fastapi import APIRouter

from app.api.kb import overview as kb_overview
from app.api.jobs import get_job_runner
from app.api.rageval import overview as rag_eval_overview

router = APIRouter()


def job_summary():
    return get_job_runner().list()


@router.get("/api/admin/overview")
def overview():
    try:
        knowledge = kb_overview()
    except Exception:
        knowledge = None
    try:
        jobs = job_summary()
    except Exception:
        jobs = None
    try:
        rag_eval = rag_eval_overview()
    except Exception:
        rag_eval = None
    return {"knowledge": knowledge, "jobs": jobs, "rag_eval": rag_eval}
