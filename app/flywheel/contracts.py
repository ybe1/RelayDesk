from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceHit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    chunk_id: int
    section_path: str
    question: str
    answer: str
    score: float | None = None
    rerank_score: float
    rank: int = Field(ge=1)
    is_key_clause: bool = False


class EvidenceSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    retrieval_id: str = Field(min_length=1, max_length=64)
    query: str = Field(min_length=1)
    strategy: str = Field(min_length=1, max_length=64)
    created_at: datetime
    hits: list[EvidenceHit]
    query_sources: list[str] = Field(default_factory=list)


class ConfidenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    threshold: float = Field(ge=0, le=1)
    top1_floor: float
    top1_ceiling: float
    gap_floor: float
    gap_ceiling: float
    effective_score_threshold: float
    top_k: int = Field(ge=1)
    calibration_id: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def valid_ranges(self):
        if self.top1_ceiling <= self.top1_floor:
            raise ValueError("top1_ceiling must exceed top1_floor")
        if self.gap_ceiling <= self.gap_floor:
            raise ValueError("gap_ceiling must exceed gap_floor")
        return self


class ConfidenceSignals(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    top1: float = Field(ge=0, le=1)
    evidence_count: float = Field(ge=0, le=1)
    gap: float = Field(ge=0, le=1)
    key_clause: float = Field(ge=0, le=1)

    @classmethod
    def zero(cls) -> "ConfidenceSignals":
        return cls(top1=0.0, evidence_count=0.0, gap=0.0, key_clause=0.0)


class ConfidenceDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    allowed: bool
    score: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    reason: Literal["evidence_ok", "no_hits", "evidence_below_threshold"]
    signals: ConfidenceSignals
