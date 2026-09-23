from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.jobs import JOBS, JobAlreadyRunning, JobRunner, JobUnavailable, UnknownJob

router = APIRouter()
_runner = JobRunner()


def get_job_runner() -> JobRunner:
    return _runner


class StartRequest(BaseModel):
    confirmation: str | None = None


def _http_job_error(error: Exception):
    if isinstance(error, UnknownJob):
        raise HTTPException(404, "unknown job") from error
    if isinstance(error, JobAlreadyRunning):
        raise HTTPException(409, "job already running") from error
    if isinstance(error, JobUnavailable):
        raise HTTPException(503, str(error)) from error
    raise error


@router.get("/api/jobs")
def jobs():
    return get_job_runner().list()


@router.post("/api/jobs/{name}", status_code=202)
def start_job(name: str, request: StartRequest | None = None):
    if name == "kb-reset" and (request is None or request.confirmation != "RESET_CH03"):
        raise HTTPException(400, "confirmation must be RESET_CH03")
    try:
        return get_job_runner().start(name)
    except (UnknownJob, JobAlreadyRunning, JobUnavailable) as error:
        _http_job_error(error)


@router.get("/api/jobs/{name}")
def job_status(name: str, limit: int = 200):
    if not 1 <= limit <= 200:
        raise HTTPException(400, "limit must be between 1 and 200")
    try:
        return get_job_runner().status(name, limit)
    except UnknownJob as error:
        _http_job_error(error)


@router.post("/api/jobs/{name}/stop")
def stop_job(name: str):
    try:
        return get_job_runner().stop(name)
    except UnknownJob as error:
        _http_job_error(error)
