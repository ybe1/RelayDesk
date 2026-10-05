from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Iterable

from .contracts import EvidenceHit, EvidenceSnapshot


def _finite_score(value: Any, name: str, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _ranked_hits(hits: Iterable[dict]) -> list[EvidenceHit]:
    normalized: list[EvidenceHit] = []
    for raw in hits:
        if not isinstance(raw, dict):
            raise TypeError("hits must contain dictionaries")
        chunk_id = raw.get("chunk_id", raw.get("id"))
        if isinstance(chunk_id, bool) or not isinstance(chunk_id, int):
            raise ValueError("chunk_id must be an integer")
        is_key_clause = raw.get("is_key_clause", False)
        if not isinstance(is_key_clause, bool):
            raise ValueError("is_key_clause must be a boolean")
        normalized.append(EvidenceHit(
            chunk_id=chunk_id,
            section_path=str(raw.get("section_path") or ""),
            question=str(raw.get("question") or ""),
            answer=str(raw.get("answer") or ""),
            score=_finite_score(raw.get("score"), "score", optional=True),
            rerank_score=_finite_score(raw.get("rerank_score"), "rerank_score"),
            rank=raw.get("rank"),
            is_key_clause=is_key_clause,
        ))
    normalized.sort(key=lambda item: item.rank)
    if [item.rank for item in normalized] != list(range(1, len(normalized) + 1)):
        raise ValueError("rank values must be unique and contiguous from 1")
    return normalized


def snapshot_hits(query: str, strategy: str, hits: Iterable[dict], *, retrieval_id: str,
                  created_at: datetime, query_sources: list[str] | None = None) -> EvidenceSnapshot:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be nonempty text")
    if not isinstance(strategy, str) or not strategy.strip():
        raise ValueError("strategy must be nonempty text")
    sources = list(dict.fromkeys(query_sources or []))
    return EvidenceSnapshot(
        retrieval_id=retrieval_id, query=query.strip(), strategy=strategy.strip(),
        created_at=created_at, hits=_ranked_hits(hits), query_sources=sources,
    )


def pool_snapshot(snapshot: EvidenceSnapshot | None, *, retrieval_completed: bool) -> list[dict] | None:
    if snapshot is None:
        if retrieval_completed:
            raise ValueError("completed retrieval requires a snapshot")
        return None
    if not retrieval_completed:
        return None
    return [hit.model_dump(mode="json") for hit in snapshot.hits[:3]]
