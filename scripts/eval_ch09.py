"""Run one complete Ch04 evaluation and persist the Ch09 trend row."""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
import argparse
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import text

from app.db.session import SessionLocal
from app.evaluation.ch09_runs import EvaluationDependencies, EvalRunRepository, run_ch09_evaluation


ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = ROOT / "data/runtime/ch09/eval-schedule.json"
REPORTS = ROOT / "data/ch09/reports"


def _schedule_state():
    try:
        return json.loads(SCHEDULE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


@contextmanager
def execution_lock():
    with SessionLocal() as session:
        acquired = session.execute(text("SELECT GET_LOCK('mewhelp_ch09_eval_execution', 0)")).scalar() == 1
        if not acquired:
            raise RuntimeError("evaluation already running")
        try:
            yield
        finally:
            session.execute(text("SELECT RELEASE_LOCK('mewhelp_ch09_eval_execution')"))


def _evaluate(path: Path):
    mode="--retry-skipped" if path.exists() else "--fresh"
    code = subprocess.run([sys.executable, "-m", "scripts.eval_ch04", mode, "--report", str(path)],
                          cwd=ROOT, check=False).returncode
    if code:
        raise RuntimeError(f"Ch04 evaluation failed with exit code {code}")
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--triggered-by",choices=("manual","scheduled"),default="manual")
    parser.add_argument("--run-id",help="Resume one explicit existing run; manual mode only")
    args=parser.parse_args(argv)
    state = _schedule_state()
    scheduled = args.triggered_by == "scheduled"
    if scheduled and not str(state.get("run_id", "")).startswith("scheduled:"):
        print("Ch09 evaluation failed: missing scheduled run identity",file=sys.stderr); return 1
    if scheduled and args.run_id:
        parser.error("--run-id is only valid for manual evaluation")
    run_id = state["run_id"] if scheduled else (args.run_id or f"manual:{uuid.uuid4().hex}")
    slot = state.get("attempted_slot") if scheduled else None
    try:
        with execution_lock():
            result = run_ch09_evaluation("定时" if scheduled else "手动", run_id,
                EvaluationDependencies(EvalRunRepository(SessionLocal), REPORTS, _evaluate, slot))
        print(json.dumps(result.metrics, ensure_ascii=False))
        if scheduled:
            state.update(status="complete")
            SCHEDULE.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
        return 0
    except Exception as error:
        if scheduled:
            state.update(status="manual_recovery_required",error=type(error).__name__)
            SCHEDULE.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
        print(f"Ch09 evaluation failed: {type(error).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
