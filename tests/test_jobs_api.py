import sys

import pytest
from fastapi.testclient import TestClient

from app.main import app


class FakeRunner:
    def __init__(self):
        self.started = []
    def list(self): return []
    def start(self, name):
        self.started.append(name)
        return {"job_id": "j1", "name": name, "state": "running", "exit_code": None, "log_tail": ""}
    def status(self, name, limit=200): return {"job_id": "j1", "name": name, "state": "succeeded", "exit_code": 0, "log_tail": "done"}
    def stop(self, name): return {"job_id": "j1", "name": name, "state": "stopped", "exit_code": None, "log_tail": ""}


def test_jobs_whitelist_and_reset_confirmation(monkeypatch):
    from app.api import jobs
    runner = FakeRunner()
    monkeypatch.setattr(jobs, "get_job_runner", lambda: runner)
    client = TestClient(app)
    assert client.post("/api/jobs/nope").status_code == 404
    assert client.post("/api/jobs/kb-reset", json={}).status_code == 400
    assert client.post("/api/jobs/kb-reset", json={"confirmation": "RESET_CH03"}).status_code == 202
    assert runner.started == ["kb-reset"]
    assert client.post("/api/jobs/kb-vectorize").status_code == 202
    assert client.get("/api/jobs/kb-vectorize").json()["log_tail"] == "done"


def test_job_commands_are_fixed_python_argv_and_never_shell(monkeypatch, tmp_path):
    from app.core.jobs import JOBS, JobRunner
    calls = []
    class Process:
        pid = 123
        def poll(self): return None
    def popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return Process()
    runner = JobRunner(project_root=tmp_path, log_dir=tmp_path, popen=popen)
    runner.start("kb-preview")
    argv, kwargs = calls[0]
    assert argv == [sys.executable, "-m", "scripts.kb", "preview"]
    assert kwargs["shell"] is False
    assert set(JOBS) >= {"kb-reset", "seed-conv", "eval-retrieval", "eval-mining"}


def test_stop_only_reports_stopped_after_the_owned_process_exits(monkeypatch, tmp_path):
    from app.core import jobs
    from app.core.jobs import JobAlreadyRunning, JobRunner

    class Process:
        pid = 321
        code = None
        def poll(self): return self.code

    process = Process()
    replacement = Process()
    processes = iter((process, replacement))
    runner = JobRunner(project_root=tmp_path, log_dir=tmp_path, popen=lambda *_args, **_kwargs: next(processes))
    runner.start("kb-preview")
    monkeypatch.setattr(jobs.os, "name", "posix")
    monkeypatch.setattr(jobs.os, "killpg", lambda _pid, _signal: None)
    assert runner.stop("kb-preview")["state"] == "running"
    with pytest.raises(JobAlreadyRunning):
        runner.start("kb-preview")
    process.code = 0
    assert runner.status("kb-preview")["state"] == "stopped"
    assert runner.start("kb-preview")["state"] == "running"


def test_stop_uses_owned_posix_process_tree_and_keeps_terminal_logs(monkeypatch, tmp_path):
    from app.core import jobs
    from app.core.jobs import JobRunner

    class Process:
        pid = 654
        code = None
        def poll(self): return self.code

    process = Process()
    runner = JobRunner(project_root=tmp_path, log_dir=tmp_path, popen=lambda *_args, **_kwargs: process)
    runner.start("kb-preview")
    record = runner._jobs["kb-preview"]
    record["log_file"].write('token: "quoted-token"\npassword = "quoted-password"\nAuthorization: Bearer top-secret\n{"api_key":"json-secret","token":"json-token"}\n')
    record["log_file"].flush()
    called = []
    monkeypatch.setattr(jobs.os, "name", "posix")
    monkeypatch.setattr(jobs.os, "killpg", lambda pid, signal: (called.append((pid, signal)), setattr(process, "code", -15)))
    stopped = runner.stop("kb-preview")
    assert stopped["state"] == "stopped"
    assert "top-secret" not in stopped["log_tail"]
    assert "json-secret" not in stopped["log_tail"]
    assert "quoted-token" not in stopped["log_tail"]
    assert "quoted-password" not in stopped["log_tail"]
    assert 'token: "[redacted]"' in stopped["log_tail"]
    assert 'password = "[redacted]"' in stopped["log_tail"]
    assert runner.tail("kb-preview", limit=1) == '{"api_key":"[redacted]","token":"[redacted]"}'
    assert called == [(654, 15)]


def test_stop_uses_owned_windows_process_tree(monkeypatch, tmp_path):
    from app.core import jobs
    from app.core.jobs import JobRunner

    class Process:
        pid = 987
        code = None
        def poll(self): return self.code

    process = Process()
    runner = JobRunner(project_root=tmp_path, log_dir=tmp_path, popen=lambda *_args, **_kwargs: process)
    runner.start("kb-preview")
    called = []
    monkeypatch.setattr(jobs.os, "name", "nt")
    monkeypatch.setattr(jobs.subprocess, "run", lambda argv, **kwargs: (called.append((argv, kwargs)), setattr(process, "code", 1)))
    assert runner.stop("kb-preview")["state"] == "stopped"
    assert called[0][0] == ["taskkill", "/PID", "987", "/T", "/F"]


def test_eval_jobs_are_rejected_until_the_task9_module_exists(monkeypatch, tmp_path):
    from app.core.jobs import JobRunner, JobUnavailable
    runner = JobRunner(project_root=tmp_path, log_dir=tmp_path, popen=lambda *_args, **_kwargs: None)
    with pytest.raises(JobUnavailable):
        runner.start("eval-retrieval")
