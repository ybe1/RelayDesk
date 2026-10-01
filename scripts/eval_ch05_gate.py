"""Calibrate a minimal score gate on labeled chapter 5 demonstration cases."""
import asyncio
import json
from pathlib import Path
from app.workflow.budget import TurnBudget
from app.workflow.retrieval import retrieve_for_turn


async def main():
    root = Path(__file__).resolve().parents[1] / "data/ch05"
    rows = []
    for line in (root / "gate_cases.jsonl").read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        result = await retrieve_for_turn(case["query"], TurnBudget())
        score = max((h["rerank_score"] for h in result.hits), default=0)
        rows.append({**case, "score": score, "hits": [h["id"] for h in result.hits]})
    low = max(r["score"] for r in rows if not r["allowed"])
    high = min(r["score"] for r in rows if r["allowed"])
    report = {"cases": rows, "separable": low < high, "threshold": (low+high)/2 if low < high else None}
    (root / "gate_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if low >= high:
        raise RuntimeError("labeled cases cannot be separated by a score threshold")


if __name__ == "__main__":
    asyncio.run(main())
