"""Run synthetic labeled prompt acceptance: python -m scripts.eval_ch04_prompts."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def assess_query(sample: dict, actual: dict) -> list[str]:
    errors = []
    if actual.get("category") != sample["category"]:
        errors.append("category")
    normalized = actual.get("normalized_query", "")
    search = actual.get("search_query", "")
    for label, text, terms in (
        ("normalized_missing", normalized, sample.get("normalized_contains", [])),
        ("search_missing", search, sample.get("search_contains", [])),
    ):
        if any(term.casefold() not in text.casefold() for term in terms):
            errors.append(label)
    if any(not any(term.casefold() in search.casefold() for term in group)
           for group in sample.get("search_any", [])):
        errors.append("search_synonym_missing")
    if any(term.casefold() in normalized.casefold() for term in sample.get("normalized_forbidden", [])):
        errors.append("normalized_synonym_leak")
    if any(term.casefold() in (normalized + " " + search).casefold() for term in sample.get("forbidden", [])):
        errors.append("forbidden_content")
    if not normalized.strip() or not search.strip():
        errors.append("blank_query")
    return errors


def assess_evidence(sample: dict, actual: dict) -> list[str]:
    errors = []
    if actual.get("useful") is not sample["useful"]:
        errors.append("useful")
    if not actual.get("reason", "").strip():
        errors.append("reason")
    return errors


def summarize(rows: list[dict]) -> dict:
    passed = sum(row["passed"] for row in rows)
    category_ok = all(row.get("category_ok", False) for row in rows if row["kind"] == "query")
    safety_ok = all(row["passed"] for row in rows if row.get("safety"))
    accuracy = passed / len(rows) if rows else 0.0
    return {"total": len(rows), "correct": passed, "accuracy": accuracy,
            "category_all_passed": category_ok, "safety_all_passed": safety_ok,
            "passed": accuracy >= 0.9 and category_ok and safety_ok}


async def evaluate() -> dict:
    from app.config import settings
    from app.core.llm import get_chat_model
    from app.knowledge.evidence_check import check_evidence
    from app.knowledge.query_understanding import understand_query

    query_path = ROOT / "tests/data/ch04_query_samples.json"
    evidence_path = ROOT / "tests/data/ch04_evidence_samples.json"
    query_data = json.loads(query_path.read_text(encoding="utf-8"))
    evidence_data = json.loads(evidence_path.read_text(encoding="utf-8"))
    # This process evaluates synthetic fixtures with their annotated exact categories.
    # Production .env and the persisted application configuration are not changed.
    original_categories = settings.rag_category_allowlist
    settings.rag_category_allowlist = tuple(query_data["category_allowlist"])
    model = get_chat_model(disable_thinking=True)
    semaphore = asyncio.Semaphore(3)

    async def one(kind, sample):
        row = {"kind": kind, "id": sample["id"], "safety": sample.get("safety", False)}
        try:
            async with semaphore:
                async with asyncio.timeout(120):
                    if kind == "query":
                        result = await understand_query(model, sample["query"])
                    else:
                        result = await check_evidence(model, sample["query"], sample["hits"])
            actual = result.model_dump()
            row["actual"] = actual
            row["errors"] = (assess_query if kind == "query" else assess_evidence)(sample, actual)
            if kind == "query":
                row["category_ok"] = actual["category"] == sample["category"]
            row["passed"] = not row["errors"]
        except Exception as exc:
            # Error bodies may carry credentials or request content; retain type only.
            row.update(passed=False, errors=[type(exc).__name__])
        return row

    try:
        rows = await asyncio.gather(
            *(one("query", sample) for sample in query_data["samples"]),
            *(one("evidence", sample) for sample in evidence_data["samples"]),
        )
    finally:
        settings.rag_category_allowlist = original_categories
    return {
        "schema": "ch04-prompt-eval/v1", "at": datetime.now(timezone.utc).isoformat(),
        "model": settings.chat_model,
        "prompt_sha256": hashlib.sha256((ROOT / "app/core/rag_prompts.py").read_bytes()).hexdigest(),
        "sample_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (query_path, evidence_path)},
        "categories": query_data["category_allowlist"],
        "summary": summarize(rows), "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(evaluate())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"]))
    print("mismatches:", ", ".join(row["id"] for row in report["rows"] if not row["passed"]) or "none")
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
