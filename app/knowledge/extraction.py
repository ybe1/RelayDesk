"""Batch extraction of reusable QA pairs from historical conversations."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from uuid import uuid4

from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db.models import Conversation, Message, QAExtractionStaging
from app.knowledge.dedup import qa_fingerprint
from app.knowledge.extraction_prompts import EXTRACTION_PROMPT


class ExtractedQA(BaseModel):
    conversation_id: int
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class QASet(BaseModel):
    items: list[ExtractedQA]


_PRIVATE_CASE_FACT = re.compile(
    r"(?:订单\s*[A-Za-z0-9_-]{2,}|(?:物流|快递|运单)\s*单号|(?:订单号|运单号)\s*[:：]?|(?:物流|快递).*(?:已|到|显示|状态|揽收|签收|派送))",
    re.IGNORECASE,
)


def _safe_messages(rows: list[Message]) -> list[dict[str, str]]:
    """Keep customer text and final textual assistant answers only."""
    messages: list[dict[str, str]] = []
    for row in rows:
        if row.role not in {"user", "assistant"} or not row.content:
            continue
        if row.role == "assistant" and row.tool_calls:
            continue
        if _PRIVATE_CASE_FACT.search(row.content):
            continue
        messages.append({"role": row.role, "content": row.content})
    return messages


def _load_batch(session_factory: Callable, conversation_ids: tuple[int, ...]) -> list[dict]:
    with session_factory() as session:
        rows = session.scalars(
            select(Message)
            .where(Message.conversation_id.in_(conversation_ids))
            .order_by(Message.conversation_id, Message.id)
        ).all()
    grouped: dict[int, list[Message]] = {conversation_id: [] for conversation_id in conversation_ids}
    for row in rows:
        grouped[row.conversation_id].append(row)
    return [
        {"conversation_id": conversation_id, "messages": messages}
        for conversation_id in conversation_ids
        if (messages := _safe_messages(grouped[conversation_id]))
    ]


def _store_batch(session_factory: Callable, batch_no: str, candidates: list[ExtractedQA], allowed_ids: set[int]) -> tuple[int, int]:
    if any(candidate.conversation_id not in allowed_ids for candidate in candidates):
        raise ValueError("extraction candidate conversation_id is outside the current batch")

    with session_factory.begin() as session:
        known_by_source: dict[str, set[str]] = {}
        for row in session.scalars(select(QAExtractionStaging)).all():
            if row.source_ref:
                known_by_source.setdefault(row.source_ref, set()).add(qa_fingerprint(row.question, row.answer))
        extracted = 0
        skipped = 0
        for candidate in candidates:
            source_ref = f"conversation/{candidate.conversation_id}"
            fingerprint = qa_fingerprint(candidate.question, candidate.answer)
            source_fingerprints = known_by_source.setdefault(source_ref, set())
            if fingerprint in source_fingerprints:
                skipped += 1
                continue
            source_fingerprints.add(fingerprint)
            session.add(QAExtractionStaging(
                batch_no=batch_no,
                source_ref=source_ref,
                question=candidate.question,
                answer=candidate.answer,
                status="extracted",
            ))
            extracted += 1
    return extracted, skipped


def extract_batches(session_factory: Callable, model, batch_size: int = 30) -> dict[str, int]:
    """Extract each batch into staging, leaving global de-duplication separate."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    with session_factory() as session:
        conversation_ids = tuple(session.scalars(select(Conversation.id).order_by(Conversation.id)).all())

    structured_model = model.with_structured_output(QASet, method="function_calling")
    totals = {"batches": 0, "extracted": 0, "skipped": 0}
    for start in range(0, len(conversation_ids), batch_size):
        conversation_batch = conversation_ids[start:start + batch_size]
        conversations = _load_batch(session_factory, conversation_batch)
        if not conversations:
            continue
        result = structured_model.invoke(EXTRACTION_PROMPT.format_messages(
            conversations=json.dumps(conversations, ensure_ascii=False)
        ))
        candidates = result.items if isinstance(result, QASet) else QASet.model_validate(result).items
        extracted, skipped = _store_batch(
            session_factory,
            uuid4().hex,
            candidates,
            set(conversation_batch),
        )
        totals["batches"] += 1
        totals["extracted"] += extracted
        totals["skipped"] += skipped
    return totals
