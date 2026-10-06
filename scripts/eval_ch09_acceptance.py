"""Produce machine-readable Ch09 local or live acceptance evidence."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


LOCAL_GROUPS = [
    ("three_pool_sources_and_snapshots", ["tests/test_ch09_workflow_gate.py", "tests/test_ch09_feedback_api.py"]),
    ("all_status_dedup", ["tests/test_ch09_flywheel_processor.py"]),
    ("approval_idempotence_and_crash_recovery", ["tests/test_ch09_review_api.py", "tests/test_ch09_incremental_indexer.py"]),
    ("trace_failure_isolation", ["tests/test_ch09_observability.py"]),
    ("scheduler_lock_and_two_fake_runs", ["tests/test_ch09_scheduler.py", "tests/test_ch09_eval_runs.py"]),
    ("review_and_observability_browser", ["tests/test_static.py"]),
]


def local_acceptance():
    checks=[]
    for name, files in LOCAL_GROUPS:
        command=[sys.executable,"-m","pytest",*files,"-q"]
        result=subprocess.run(command,text=True,capture_output=True,check=False)
        checks.append({"name":name,"passed":result.returncode==0,"command":" ".join(command),
                       "summary":(result.stdout+result.stderr).strip().splitlines()[-1:]})
    browser=subprocess.run([sys.executable,"-m","scripts.verify_ch09_browser","--base-url","http://127.0.0.1:8000",
                            "--output","data/ch09/reports/browser.json"],text=True,capture_output=True,check=False)
    checks.append({"name":"headless_browser_states","passed":browser.returncode==0,
                   "command":"python -m scripts.verify_ch09_browser", "summary":browser.stdout.strip().splitlines()[-1:]})
    return {"schema":"ch09-acceptance/v1","mode":"local","created_at":datetime.now(UTC).isoformat(),
            "passed":all(item["passed"] for item in checks),"checks":checks,
            "provider_calls":{"deepseek":0,"siliconflow":0,"langfuse_local":0}}


def live_acceptance():
    raise RuntimeError("live acceptance must be run interactively after explicit Chapter 9 cloud-data authorization")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--mode",choices=("local","live"),required=True);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    try: report=local_acceptance() if args.mode=="local" else live_acceptance()
    except Exception as error:
        report={"schema":"ch09-acceptance/v1","mode":args.mode,"created_at":datetime.now(UTC).isoformat(),
                "passed":False,"blocked":type(error).__name__,"reason":str(error),"provider_calls":{}}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(report,ensure_ascii=False));return 0 if report.get("passed") else 1


if __name__=="__main__": raise SystemExit(main())
