"""Replay saved faith-case snapshots: exit 0 pass, 1 mismatch, 2 incomplete."""

import argparse
import asyncio
import json

from app.evaluation.faithfulness import EmptyCompletion, judge_faithfulness

REVIEWED_STATUSES = ("已解决", "无需解决")


async def replay_cases(cases, model):
    results = []
    for case in cases:
        if case["status"] == "未解决":
            results.append({"id": case["id"], "eval_id": case["eval_id"],
                            "status": "skipped", "reason": "not_reviewed"})
            continue
        row = {"id": case["id"], "eval_id": case["eval_id"],
               "expected": case["status"] == "无需解决"}
        if case["status"] not in REVIEWED_STATUSES or case["citations"] is None:
            results.append({**row, "status": "excluded", "reason": "missing_snapshot_or_invalid_status"})
            continue
        for attempt in range(2):
            try:
                async with asyncio.timeout(90):
                    judgment = await judge_faithfulness(model, case["query"], case["answer"], case["citations"])
                results.append({**row, "status": "checked", "actual": judgment.faithful,
                                "match": judgment.faithful == row["expected"], "reason": judgment.reason})
                break
            except Exception as error:
                if isinstance(error, EmptyCompletion) and attempt == 0:
                    continue
                results.append({**row, "status": "excluded", "reason": type(error).__name__})
                break
    checked = [row for row in results if row["status"] == "checked"]
    mismatches = sum(not row["match"] for row in checked)
    excluded = sum(row["status"] == "excluded" for row in results)
    code = 1 if mismatches else (2 if excluded or not checked else 0)
    return {"status": "mismatch" if code == 1 else ("incomplete" if code == 2 else "passed"),
            "checked": len(checked), "excluded": excluded, "mismatches": mismatches,
            "agreement_rate": (len(checked) - mismatches) / len(checked) if checked else None,
            "exit_code": code, "cases": results}


def load_cases():
    from sqlalchemy import select
    from app.db.models import FaithCase
    from app.db.session import SessionLocal
    with SessionLocal() as session:
        return [{key: getattr(row, key) for key in
                 ("id", "eval_id", "query", "answer", "citations", "status", "resolution")}
                for row in session.scalars(select(FaithCase).order_by(FaithCase.id))]


def get_model():
    from app.core.llm import get_chat_model
    return get_chat_model(disable_thinking=True, temperature=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-only", action="store_true", help="Read ledger counts without calling the model")
    args = parser.parse_args(argv)
    try:
        cases = load_cases()
        reviewed = sum(case["status"] in REVIEWED_STATUSES for case in cases)
        if args.inspect_only or not reviewed:
            result = {"status": "blocked" if args.inspect_only else "incomplete",
                      "reason": "cloud_calls_disabled" if args.inspect_only else "no_reviewed_cases",
                      "total": len(cases), "reviewed": reviewed,
                      "exit_code": 2}
        else:
            result = asyncio.run(replay_cases(cases, get_model()))
    except Exception as error:
        result = {"status": "blocked", "reason": type(error).__name__, "exit_code": 2}
    print(json.dumps(result, ensure_ascii=False))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
