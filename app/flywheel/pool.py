from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import select, text

from app.db.models import LowConfidenceQuestion
from app.repositories.low_confidence import insert_low_confidence


@dataclass(frozen=True)
class GapEvent:
    conversation_id: int | None
    raw_question: str
    source: Literal["retrieval_low_conf", "self_check", "user_feedback"]
    reason: str
    retrieved_chunks: list[dict] | dict | None
    event_key: str


def _digest(event_key: str) -> str:
    if not event_key or len(event_key) > 256:
        raise ValueError("invalid gap event key")
    return hashlib.sha256(event_key.encode("utf-8")).hexdigest()


def gap_lock_name(event_key: str) -> str:
    """MySQL advisory lock names are limited to 64 characters."""
    return f"mewhelp:gap:{_digest(event_key)[:48]}"


def record_gap(event: GapEvent, session_factory) -> int:
    digest = _digest(event.event_key)
    marker = f"[event:{digest}]"
    lock_name = gap_lock_name(event.event_key)
    with session_factory() as session:
        mysql = session.bind is not None and session.bind.dialect.name == "mysql"
        acquired = False
        try:
            if mysql:
                acquired = session.execute(text("SELECT GET_LOCK(:name, 5)"), {"name": lock_name}).scalar() == 1
                if not acquired:
                    raise TimeoutError("cannot acquire gap event lock")
            existing = session.scalar(select(LowConfidenceQuestion).where(
                LowConfidenceQuestion.reason.like(f"%{marker}%")
            ).with_for_update())
            if existing is not None:
                return existing.id
            reason = f"{event.reason}\n{marker}" if event.reason else marker
            row = insert_low_confidence(
                session, event.conversation_id, event.raw_question, event.source, reason,
                retrieved_chunks=event.retrieved_chunks,
            )
            session.commit()
            return row.id
        except Exception:
            session.rollback()
            raise
        finally:
            if mysql and acquired:
                session.execute(text("SELECT RELEASE_LOCK(:name)"), {"name": lock_name})
