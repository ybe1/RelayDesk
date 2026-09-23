from fastapi import APIRouter

from app.api.kb import overview as kb_overview
from app.api.jobs import get_job_runner

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
    return {"knowledge": knowledge, "jobs": jobs}
