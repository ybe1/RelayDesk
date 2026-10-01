"""Validate labels offline; evaluate the actual classifier without --offline."""
import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path
from app.workflow.intent import IntentDecision, classify_intent, detect_smalltalk

DATA = Path(__file__).resolve().parents[1] / "data/ch05/intent_cases.jsonl"


async def main(offline):
    cases = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    for case in cases:
        IntentDecision(intent=case["intent"])
        if detect_smalltalk(case["query"]) and case["intent"] != "闲聊":
            raise ValueError("smalltalk rule swallowed a business request")
    counts = Counter(case["intent"] for case in cases)
    if len(counts) != 7 or min(counts.values()) < 4:
        raise ValueError("need at least four labels per intent")
    if offline:
        print(json.dumps({"mode": "dataset_validation_only", "cases": len(cases), "labels": counts}, ensure_ascii=False))
        return
    from app.core.llm import get_chat_model
    from app.workflow.budget import TurnBudget
    from app.workflow.model import BudgetedModel
    model = get_chat_model(disable_thinking=True, temperature=0).model_copy(update={"max_retries": 0})
    errors = []
    for case in cases:
        try:
            result = await classify_intent(BudgetedModel(model, TurnBudget()), case["query"])
            actual = result.intent
        except Exception as exc:
            actual = type(exc).__name__
        if actual != case["intent"]:
            errors.append({**case, "actual": actual})
    report = {"mode": "live", "cases": len(cases), "accuracy": (len(cases)-len(errors))/len(cases), "errors": errors}
    out = DATA.parent / "intent_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    asyncio.run(main(parser.parse_args().offline))
