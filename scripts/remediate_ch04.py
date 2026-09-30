"""Replay explicitly selected stored snapshots without retrieval or ledger writes.

Run with one --case-id per authorized case. Every run creates a separate artifact
in the source report's remediation/ directory; full reports are read only.
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4

from app.evaluation.ch04_metrics import answer_point_coverage, normalize
from app.evaluation.faithfulness import (
    FaithfulnessDecision, GeneratedAnswer, citation_snapshot, generate_answer, judge_faithfulness,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/ch04/reports/rag_eval.json"
AUTHORIZED_CASE_IDS = frozenset({"ch04-C-053", "ch04-E-041", "ch04-E-042"})


async def remediate(source, case_ids, generator, judge, *, model_name, timeout=120):
    source = Path(source).resolve()
    if (not case_ids or len(set(case_ids)) != len(case_ids) or timeout <= 0
            or not set(case_ids).issubset(AUTHORIZED_CASE_IDS)):
        raise ValueError("select unique explicit case IDs and a positive timeout")
    original = source.read_bytes()
    data = json.loads(original)
    available = data["cases"]["hybrid_rerank"]
    selected = []
    for case_id in case_ids:
        matches = [row for row in available if row["id"] == case_id]
        if len(matches) != 1 or matches[0]["retrieval"]["status"] != "ok":
            raise ValueError("selected case requires exactly one completed retrieval snapshot")
        row = matches[0]
        if row["bucket"] == "D_out_of_scope" or not row["expected_groups"]:
            raise ValueError("remediation requires labeled answer points")
        selected.append(row)
    output_dir = source.parent / "remediation"
    # Reject symlink/junction redirects before sending any data or writing outputs.
    if output_dir.resolve() != output_dir:
        raise ValueError("remediation directory must not redirect elsewhere")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema": "ch04-remediation/v1", "at": datetime.now(timezone.utc).isoformat(),
              "strategy": "hybrid_rerank", "model": model_name,
              "source_sha256": hashlib.sha256(original).hexdigest(),
              "prompt_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ("app/evaluation/faithfulness.py", "app/core/rag_prompts.py")},
              "metric_note": "Whitespace-normalized literal labeled points; semantic review is separate.",
              "rows": []}
    for case in selected:
        citations = citation_snapshot(case["retrieval"]["hits"])
        row = {"id": case["id"], "query": case["query"], "expected_groups": case["expected_groups"],
               "retrieval": deepcopy(case["retrieval"]), "citations": citations,
               "original_generation": case["generation"], "original_faithfulness": case["faithfulness"],
               "passed": False}
        report["rows"].append(row)
        try:
            result = await asyncio.wait_for(generator(case["query"], deepcopy(case["retrieval"])), timeout)
            answer = GeneratedAnswer.model_validate(result)
            coverage = answer_point_coverage(case, answer.answer)
            row["generation"] = {**answer.model_dump(), "answer_point_coverage": coverage}
            row["answer_points"] = [{"point": point, "covered": normalize(point) in normalize(answer.answer)}
                                    for group in case["expected_groups"] for point in group["expect_points"]]
            decision = FaithfulnessDecision.model_validate(await asyncio.wait_for(
                judge(case["query"], answer.answer, citations), timeout))
            row["faithfulness"] = decision.model_dump()
            row["passed"] = decision.faithful and not answer.refused and coverage == 1.0
        except Exception as error:
            # Upstream exception text may contain credentials or request contents.
            row["error"] = type(error).__name__
    report["source_unchanged"] = source.read_bytes() == original
    passed = sum(row["passed"] for row in report["rows"])
    report["summary"] = {"total": len(selected), "passed": passed,
                         "all_passed": passed == len(selected) and report["source_unchanged"]}
    output = output_dir / f"targeted-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}.json"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report, output


async def _live(args):
    from app.config import settings
    from app.core.llm import get_chat_model
    model = get_chat_model(disable_thinking=True)
    judge_model = get_chat_model(disable_thinking=True, temperature=0)
    # No implicit SDK retries; reruns remain explicit and retain separate evidence.
    model.max_retries = judge_model.max_retries = 0

    async def generator(query, retrieval):
        return await generate_answer(model, query, retrieval)

    async def judge(query, answer, citations):
        return await judge_faithfulness(judge_model, query, answer, citations)

    return await remediate(args.source, args.case_id, generator, judge,
                           model_name=settings.chat_model, timeout=args.timeout)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args(argv)
    try:
        report, output = asyncio.run(_live(args))
    except (OSError, ValueError, KeyError) as error:
        print(f"Remediation could not start or save: {type(error).__name__}", file=sys.stderr)
        return 2
    print(json.dumps(report["summary"]))
    print(output)
    return 0 if report["summary"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
