from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.flywheel.confidence import WEIGHTS
from app.flywheel.contracts import ConfidenceConfig


class NoSafeUsableThreshold(ValueError):
    """No candidate rejects every negative case while allowing a positive case."""


class ConfidenceConfigurationError(RuntimeError):
    """A frozen calibration report is incompatible with the live retriever."""


class ScoredCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    case_id: str = Field(min_length=1)
    fact_group: str = Field(min_length=1)
    expected_allowed: bool
    score: float = Field(ge=0, le=1, allow_inf_nan=False)


class CalibrationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    false_pass_rate: float = Field(ge=0, le=1)
    false_block_rate: float = Field(ge=0, le=1)
    false_pass_count: int = Field(ge=0)
    false_block_count: int = Field(ge=0)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    f_0_5: float = Field(ge=0, le=1)
    confusion: dict[str, int]
    weights: dict[str, float]
    dataset_hash: str = Field(min_length=1)
    corpus_hash: str = Field(min_length=1)
    rerank_model: str = Field(min_length=1)
    top_k: int = Field(ge=1)
    top1_floor: float = 0.0
    top1_ceiling: float = 1.0
    gap_floor: float = 0.0
    gap_ceiling: float = 1.0
    effective_score_threshold: float = 0.5
    calibration_id: str = Field(min_length=1)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f_score(precision: float, recall: float, beta: float) -> float:
    denominator = beta * beta * precision + recall
    return (1 + beta * beta) * precision * recall / denominator if denominator else 0.0


def _evaluate(rows: Sequence[ScoredCase], threshold: float) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in rows:
        predicted = row.score >= threshold
        if predicted and row.expected_allowed:
            tp += 1
        elif predicted:
            fp += 1
        elif row.expected_allowed:
            fn += 1
        else:
            tn += 1
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return {
        "threshold": threshold,
        "false_pass_rate": _ratio(fp, fp + tn),
        "false_block_rate": _ratio(fn, tp + fn),
        "false_pass_count": fp,
        "false_block_count": fn,
        "precision": precision,
        "recall": recall,
        "f1": _f_score(precision, recall, 1.0),
        "f_0_5": _f_score(precision, recall, 0.5),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def select_threshold(
    rows: Sequence[ScoredCase],
    *,
    candidate_thresholds: Iterable[float],
    metadata: Mapping[str, Any],
) -> CalibrationReport:
    if not rows or not any(row.expected_allowed for row in rows) or not any(not row.expected_allowed for row in rows):
        raise ValueError("calibration requires both answerable and refusal cases")
    if len({row.case_id for row in rows}) != len(rows):
        raise ValueError("calibration case_id values must be unique")

    candidates: list[dict[str, Any]] = []
    for raw_threshold in candidate_thresholds:
        threshold = float(raw_threshold)
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("candidate thresholds must be finite values from 0 to 1")
        result = _evaluate(rows, threshold)
        if result["false_pass_count"] == 0 and result["confusion"]["tp"] > 0:
            candidates.append(result)
    if not candidates:
        raise NoSafeUsableThreshold("no candidate achieved zero false passes without refusing every answerable case")

    chosen = min(candidates, key=lambda item: (item["false_block_count"], -item["threshold"]))
    required = ("dataset_hash", "corpus_hash", "rerank_model", "top_k")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"missing calibration metadata: {', '.join(missing)}")
    calibration_id = str(metadata.get("calibration_id") or f"{metadata['dataset_hash']}{metadata['corpus_hash']}")
    return CalibrationReport(
        **chosen,
        weights=dict(WEIGHTS),
        dataset_hash=str(metadata["dataset_hash"]),
        corpus_hash=str(metadata["corpus_hash"]),
        rerank_model=str(metadata["rerank_model"]),
        top_k=int(metadata["top_k"]),
        top1_floor=float(metadata.get("top1_floor", 0.0)),
        top1_ceiling=float(metadata.get("top1_ceiling", 1.0)),
        gap_floor=float(metadata.get("gap_floor", 0.0)),
        gap_ceiling=float(metadata.get("gap_ceiling", 1.0)),
        effective_score_threshold=float(metadata.get("effective_score_threshold", 0.5)),
        calibration_id=calibration_id,
    )


def save_report(report: CalibrationReport, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def load_confidence_config(
    path: str | Path, *, corpus_hash: str, rerank_model: str, top_k: int,
) -> ConfidenceConfig:
    try:
        report = CalibrationReport.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfidenceConfigurationError(f"cannot load confidence calibration report: {exc}") from exc
    actual = (corpus_hash, rerank_model, top_k)
    expected = (report.corpus_hash, report.rerank_model, report.top_k)
    if actual != expected:
        raise ConfidenceConfigurationError(
            "confidence calibration does not match the active corpus, rerank model, or Top-K"
        )
    return ConfidenceConfig(
        threshold=report.threshold,
        top1_floor=report.top1_floor,
        top1_ceiling=report.top1_ceiling,
        gap_floor=report.gap_floor,
        gap_ceiling=report.gap_ceiling,
        effective_score_threshold=report.effective_score_threshold,
        top_k=report.top_k,
        calibration_id=report.calibration_id,
    )
