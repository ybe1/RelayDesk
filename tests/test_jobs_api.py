import sys

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
