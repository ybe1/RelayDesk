"""Local offline/real-MCP verification. --live is intentionally gated separately."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'data/ch08/reports'


def save(report,name):
    REPORTS.mkdir(parents=True,exist_ok=True)
    (REPORTS/name).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


def run_local():
    migration=subprocess.run([sys.executable,'-m','scripts.migrate_ch08','--check'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    if migration.returncode:
        return {'mode':'local','status':'failed','migration_check':migration.stdout+ migration.stderr,'returncode':migration.returncode}
    suites=['tests/test_ch08_registry.py','tests/test_ch08_audit.py','tests/test_ch08_executor.py',
        'tests/test_ch08_mcp.py','tests/test_ch08_ticket_confirmation.py','tests/test_ch08_ticket_workflow.py',
        'tests/test_ch08_confirmation_api.py','tests/test_ch08_context_budget.py','tests/test_ch08_integration.py',
        'tests/test_ch08_history_redaction.py']
    tests=subprocess.run([sys.executable,'-m','pytest','-q',*suites],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    return {'mode':'local','model_mode':'offline substitutes + actual local MCP subprocesses + test MySQL',
        'status':'passed' if tests.returncode==0 else 'failed','migration_check':migration.stdout.strip(),
        'pytest_returncode':tests.returncode,'pytest_summary':tests.stdout.splitlines()[-1] if tests.stdout else '',
        'pytest_output':(tests.stdout+'\n'+tests.stderr)[-12000:],
        'cases':[json.loads(line) for line in (ROOT/'data/ch08/tool_cases.jsonl').read_text(encoding='utf-8-sig').splitlines() if line],
        'timestamp_utc':datetime.now(timezone.utc).isoformat()}


def main():
    parser=argparse.ArgumentParser(); mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--local',action='store_true'); mode.add_argument('--live',action='store_true')
    args=parser.parse_args()
    if args.live:
        report={'mode':'live','status':'blocked','reason':'本机模型数据外发授权尚未确认；未发送任何数据。'}
        save(report,'live.json');print(json.dumps(report,ensure_ascii=False,indent=2));return 2
    report=run_local();save(report,'local.json');print(json.dumps({k:v for k,v in report.items() if k!='pytest_output'},ensure_ascii=False,indent=2));return report.get('returncode',report.get('pytest_returncode',0))


if __name__=='__main__':raise SystemExit(main())
