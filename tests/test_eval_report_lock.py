"""Independent job processes must not read/overwrite a running evaluation."""

import json
from pathlib import Path
import subprocess
import sys
import time

import pytest


DRIVER = """
import asyncio
from pathlib import Path
import sys
from scripts import eval_ch04
from tests.test_eval_ch04_runner import cases, retrieve

report, entered, release = map(Path, sys.argv[1:4])
eval_ch04.validate_records = lambda rows: []
async def live(args, rows, resume):
    entered.touch()
    while not release.exists():
        await asyncio.sleep(0.01)
    return await eval_ch04.evaluate(cases(), retriever=retrieve, report_path=args.report, phase='retrieval')
eval_ch04._live = live
raise SystemExit(eval_ch04.main(['--report', str(report), '--phase', 'retrieval', *sys.argv[4:]]))
"""


def wait_until(condition):
    deadline = time.monotonic() + 15
    while not condition():
        if time.monotonic() >= deadline:
            pytest.fail("evaluation process did not reach the expected state")
        time.sleep(0.02)


@pytest.mark.parametrize("mode", ["resume", "retry", "fresh"])
def test_independent_job_runners_share_report_lock(tmp_path, mode):
    from app.core.jobs import JobRunner, PROJECT_ROOT

    path = tmp_path / "report.json"
    original = json.dumps({"existing": "checkpoint"}).encode()
    path.write_bytes(original)
    first_entered, second_entered = tmp_path / "first", tmp_path / "second"
    release = tmp_path / "release"
    second_release = tmp_path / "second-release"
    second_release.touch()
    processes = []
    def runner(label, entered, released):
        def popen(argv, **kwargs):
            process = subprocess.Popen([sys.executable, "-B", "-c", DRIVER, str(path),
                                        str(entered), str(released), *argv[3:]], **kwargs)
            processes.append(process)
            return process
        return JobRunner(project_root=PROJECT_ROOT, log_dir=tmp_path / label, popen=popen)
    first = runner("first-logs", first_entered, release)
    second = runner("second-logs", second_entered, second_release)
    try:
        first.start("eval-rag")
        wait_until(first_entered.exists)
        second.start("eval-rag", mode=mode)
        wait_until(lambda: second.status("eval-rag")["state"] != "running")
        busy = second.status("eval-rag")
        assert busy["exit_code"] == 3
        assert "busy" in busy["log_tail"].lower()
        assert not second_entered.exists()
        assert path.read_bytes() == original
        release.touch()
        wait_until(lambda: first.status("eval-rag")["state"] != "running")
        assert first.status("eval-rag")["exit_code"] == 0
        second.start("eval-rag", mode=mode)
        wait_until(lambda: second.status("eval-rag")["state"] != "running")
        assert second.status("eval-rag")["exit_code"] == 0
        assert second_entered.exists()
    finally:
        release.touch()
        for process in processes:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=10)
        first.status("eval-rag")
        second.status("eval-rag")
