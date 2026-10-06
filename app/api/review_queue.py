from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import func, select

from app.db.models import KnowledgeChunk, LowConfidenceQuestion, ReviewQueue
from app.flywheel.review import SOURCE_PREFIX, ReviewConflict, approve_review, reject_review


router = APIRouter()


def get_review_sessions():
    from app.db.session import SessionLocal
    return SessionLocal


def _factory(request: Request):
    return getattr(request.app.state, "review_session_factory", None) or get_review_sessions()


def _summary(row: ReviewQueue) -> dict:
    return {"id": row.id, "normalized_question": row.normalized_question,
            "ai_suggested_answer": row.ai_suggested_answer, "occurrence_count": row.occurrence_count,
            "review_status": row.review_status, "approved_answer": row.approved_answer,
            "created_at": row.created_at, "updated_at": row.updated_at}


@router.get("/api/review-queue")
def list_review_queue(request: Request, status: Literal["待审", "通过", "驳回"] = "待审",
                      page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100)):
    factory = _factory(request)
    with factory() as session:
        condition = ReviewQueue.review_status == status
        total = session.scalar(select(func.count()).select_from(ReviewQueue).where(condition)) or 0
        rows = session.scalars(select(ReviewQueue).where(condition).order_by(
            ReviewQueue.occurrence_count.desc(), ReviewQueue.id).offset((page - 1) * page_size).limit(page_size))
        return {"items": [_summary(row) for row in rows], "total": total, "page": page, "page_size": page_size}


@router.get("/api/review-queue/{review_id}")
def review_detail(request: Request, review_id: int):
    factory = _factory(request)
    with factory() as session:
        review = session.get(ReviewQueue, review_id)
        if review is None:
            raise HTTPException(404, "待审问题不存在")
        raw = session.scalars(select(LowConfidenceQuestion).where(
            LowConfidenceQuestion.matched_review_id == review_id).order_by(LowConfidenceQuestion.id)).all()
        chunk = session.scalar(select(KnowledgeChunk).where(
            KnowledgeChunk.section_path == f"{SOURCE_PREFIX}{review_id}"))
        return {**_summary(review), "vectorize_status": None if chunk is None else chunk.vectorize_status,
                "chunk_id": None if chunk is None else chunk.id,
                "raw_questions": [{"id": row.id, "raw_question": row.raw_question, "source": row.source,
                    "reason": row.reason, "retrieved_chunks": row.retrieved_chunks,
                    "conversation_id": row.conversation_id, "created_at": row.created_at} for row in raw]}


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    decision: Literal["通过", "驳回"]
    expected_status: Literal["待审", "通过", "驳回"] = "待审"
    approved_answer: str | None = Field(default=None)

    @model_validator(mode="after")
    def require_answer(self):
        if self.decision == "通过" and (not self.approved_answer or not self.approved_answer.strip()):
            raise ValueError("通过时必须填写核准答案")
        return self


@router.post("/api/review-queue/{review_id}/decision")
async def decide_review(request: Request, review_id: int, payload: DecisionRequest):
    factory = _factory(request)
    try:
        if payload.decision == "通过":
            result = await asyncio.to_thread(approve_review, factory, review_id,
                payload.approved_answer or "", payload.expected_status)
        else:
            result = await asyncio.to_thread(reject_review, factory, review_id, payload.expected_status)
    except LookupError as exc:
        raise HTTPException(404, "待审问题不存在") from exc
    except ReviewConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return result
