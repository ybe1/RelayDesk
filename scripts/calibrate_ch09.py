from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from app.evaluation.confidence_calibration import ScoredCase, save_report, select_threshold


DEFAULT_MANIFEST = Path("data/ch09/confidence_cases.jsonl")
DEFAULT_REPORT = Path("data/ch09/reports/confidence-calibration.json")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select and freeze the Ch09 evidence threshold")
    parser.add_argument("--scores", type=Path, required=True, help="JSONL rows with case_id/fact_group/expected_allowed/score")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--corpus-hash", required=True)
    parser.add_argument("--rerank-model", required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    rows = [ScoredCase.model_validate(row) for row in _jsonl(args.scores)]
    candidates = sorted({0.0, 1.0, *(row.score for row in rows)})
    report = select_threshold(
        rows,
        candidate_thresholds=candidates,
        metadata={
            "dataset_hash": _sha256(args.manifest),
            "corpus_hash": args.corpus_hash,
            "rerank_model": args.rerank_model,
            "top_k": args.top_k,
        },
    )
    save_report(report, args.output)
    print(json.dumps(report.model_dump(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
