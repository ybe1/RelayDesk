from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from .contracts import ConfidenceConfig, ConfidenceDecision, ConfidenceSignals, EvidenceHit


WEIGHTS = {"top1": 0.5, "evidence_count": 0.2, "gap": 0.2, "key_clause": 0.1}


class InvalidConfidenceInput(ValueError):
    pass


def _as_mapping(hit: EvidenceHit | Mapping[str, Any]) -> Mapping[str, Any]:
    return hit.model_dump() if isinstance(hit, BaseModel) else hit


def _score(hit: Mapping[str, Any]) -> float:
    value = hit.get("rerank_score")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise InvalidConfidenceInput("rerank_score must be a finite number")
    return float(value)


def _unit(value: float, floor: float, ceiling: float) -> float:
    return min(1.0, max(0.0, (value - floor) / (ceiling - floor)))


def compute_confidence(hits: Sequence[EvidenceHit | Mapping[str, Any]],
                       config: ConfidenceConfig) -> ConfidenceDecision:
    if not hits:
        return ConfidenceDecision(
            allowed=False, score=0.0, threshold=config.threshold,
            reason="no_hits", signals=ConfidenceSignals.zero(),
        )
    ranked = sorted((_as_mapping(hit) for hit in hits), key=lambda hit: hit.get("rank", 0))
    ranks = [hit.get("rank") for hit in ranked]
    if any(isinstance(rank, bool) or not isinstance(rank, int) for rank in ranks) \
            or ranks != list(range(1, len(ranked) + 1)):
        raise InvalidConfidenceInput("rank values must be unique and contiguous from 1")
    scores = [_score(hit) for hit in ranked]
    top1 = _unit(scores[0], config.top1_floor, config.top1_ceiling)
    gap = 0.0 if len(scores) == 1 else _unit(scores[0] - scores[1], config.gap_floor, config.gap_ceiling)
    sections = {
        str(hit.get("section_path") or f"chunk:{hit.get('chunk_id', hit.get('id'))}")
        for hit, score in zip(ranked, scores, strict=True)
        if score >= config.effective_score_threshold
    }
    evidence_count = min(len(sections), config.top_k) / config.top_k
    key_clause = float(any(hit.get("is_key_clause") is True for hit in ranked))
    signals = ConfidenceSignals(
        top1=top1, evidence_count=evidence_count, gap=gap, key_clause=key_clause,
    )
    score = sum(WEIGHTS[name] * getattr(signals, name) for name in WEIGHTS)
    allowed = score >= config.threshold
    return ConfidenceDecision(
        allowed=allowed, score=score, threshold=config.threshold,
        reason="evidence_ok" if allowed else "evidence_below_threshold",
        signals=signals,
    )
