from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, text

from app.db.models import LowConfidenceQuestion, ReviewQueue
from app.flywheel.prompts import FLYWHEEL_PROMPT
from app.flywheel.locks import FLYWHEEL_LOCK_NAME, LOCAL_FLYWHEEL_LOCK
from app.repositories.review_queue import list_all_candidates, pending_gaps


class FlywheelDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    normalized_question: str = Field(min_length=2, max_length=512)
    ai_suggested_answer: str | None = None
    matched_question_id: int | None = None


class CandidateBudgetError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProcessReport:
    scanned: int = 0
    created: int = 0
    merged: int = 0
    skipped: int = 0
    model_calls: int = 0


_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")


def _deidentify(value: str) -> str:
    return _PHONE.sub("[手机号]", value)


def _atomic_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _operational(row: LowConfidenceQuestion) -> bool:
    reason = row.reason or ""
    return any(marker in reason for marker in ("retrieval_unavailable:", "运行故障", "服务暂时不可用"))


async def process_pending(*, session_factory, model, batch_size: int = 20,
                          state_path: str | Path = "data/runtime/ch09/flywheel-state.json",
                          max_candidate_chars: int = 50_000) -> ProcessReport:
    if not 1 <= batch_size <= 200:
        raise ValueError("batch_size must be between 1 and 200")
    lock_session = session_factory()
    mysql = lock_session.bind is not None and lock_session.bind.dialect.name == "mysql"
    acquired = False
    try:
        if mysql:
            acquired = lock_session.execute(text("SELECT GET_LOCK(:name, 0)"), {"name": FLYWHEEL_LOCK_NAME}).scalar() == 1
        else:
            acquired = LOCAL_FLYWHEEL_LOCK.acquire(blocking=False)
        if not acquired:
            return ProcessReport()
        with session_factory() as session:
            rows = pending_gaps(session, batch_size)
            candidates = list_all_candidates(session)
            candidate_payload = [{"id": row.id, "question": row.normalized_question,
                                  "status": row.review_status} for row in candidates]
        serialized_candidates = json.dumps(candidate_payload, ensure_ascii=False, separators=(",", ":"))
        if len(serialized_candidates) > max_candidate_chars:
            raise CandidateBudgetError("review candidates exceed the single-call context budget")
        allowed_ids = {row["id"] for row in candidate_payload}
        counts = {"scanned": len(rows), "created": 0, "merged": 0, "skipped": 0, "model_calls": 0}
        skipped_ids = []
        for detached in rows:
            if _operational(detached):
                counts["skipped"] += 1
                skipped_ids.append(detached.id)
                continue
            payload = json.dumps({"question": _deidentify(detached.raw_question),
                                  "candidates": candidate_payload}, ensure_ascii=False)
            messages = FLYWHEEL_PROMPT.format_messages(payload=payload)
            raw = await model.with_structured_output(FlywheelDecision, method="function_calling").ainvoke(messages)
            decision = FlywheelDecision.model_validate(raw.model_dump() if isinstance(raw, BaseModel) else raw)
            counts["model_calls"] += 1
            if decision.matched_question_id is not None and decision.matched_question_id not in allowed_ids:
                raise ValueError("model returned a candidate ID that was not supplied")
            with session_factory() as session:
                gap = session.scalar(select(LowConfidenceQuestion).where(
                    LowConfidenceQuestion.id == detached.id).with_for_update())
                if gap is None or gap.matched_review_id is not None:
                    continue
                if decision.matched_question_id is None:
                    review = ReviewQueue(normalized_question=decision.normalized_question,
                        ai_suggested_answer=decision.ai_suggested_answer)
                    session.add(review); session.flush()
                    candidate_payload.append({"id": review.id, "question": review.normalized_question,
                                              "status": review.review_status})
                    allowed_ids.add(review.id)
                    counts["created"] += 1
                else:
                    review = session.scalar(select(ReviewQueue).where(
                        ReviewQueue.id == decision.matched_question_id).with_for_update())
                    if review is None:
                        raise ValueError("matched review candidate disappeared")
                    review.occurrence_count += 1
                    counts["merged"] += 1
                gap.matched_review_id = review.id
                session.commit()
        if skipped_ids:
            await asyncio.to_thread(_atomic_state, Path(state_path), {"operational_skips": skipped_ids})
        return ProcessReport(**counts)
    finally:
        if mysql and acquired:
            lock_session.execute(text("SELECT RELEASE_LOCK(:name)"), {"name": FLYWHEEL_LOCK_NAME})
        elif acquired:
            LOCAL_FLYWHEEL_LOCK.release()
        lock_session.close()
