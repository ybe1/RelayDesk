import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from threading import Event

import pytest
from fastapi.testclient import TestClient

from app.main import app


class FakeRunner:
    def __init__(self):
        self.started = []
    def list(self): return []
    def start(self, name, *, mode=None):
        self.started.append(name)
        return {"job_id": "j1", "name": name, "state": "running", "exit_code": None, "log_tail": ""}
    def status(self, name, limit=200): return {"job_id": "j1", "name": name, "state": "succeeded", "exit_code": 0, "log_tail": "done"}
    def stop(self, name): return {"job_id": "j1", "name": name, "state": "stopped", "exit_code": None, "log_tail": ""}


def test_jobs_whitelist_and_reset_confirmation(monkeypatch):
    from app.api import jobs
    from app.config import settings
    monkeypatch.setattr(settings, "milvus_collection", "knowledge")
    runner = FakeRunner()
    monkeypatch.setattr(jobs, "get_job_runner", lambda: runner)
    client = TestClient(app)
    assert client.post("/api/jobs/nope").status_code == 404
    assert client.post("/api/jobs/kb-reset", json={}).status_code == 400
    assert client.post("/api/jobs/kb-reset", json={"confirmation": "RESET_CH03"}).status_code == 202
    assert runner.started == ["kb-reset"]
    assert client.post("/api/jobs/kb-vectorize").status_code == 202
    assert client.get("/api/jobs/kb-vectorize").json()["log_tail"] == "done"


def test_legacy_reset_is_blocked_by_server_after_ch04_cutover(monkeypatch):
    from app.api import jobs
    from app.config import settings

    runner = FakeRunner()
    monkeypatch.setattr(jobs, "get_job_runner", lambda: runner)
    monkeypatch.setattr(settings, "milvus_collection", "knowledge_ch04")
    response = TestClient(app).post("/api/jobs/kb-reset", json={"confirmation": "RESET_CH03"})
    assert response.status_code == 409
    assert "kb-reset is disabled" in response.json()["detail"]
    assert runner.started == []


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
    monkeypatch.setattr(jobs.os, "killpg", lambda _pid, _signal: None, raising=False)
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
    monkeypatch.setattr(jobs.os, "killpg", lambda pid, signal: (called.append((pid, signal)), setattr(process, "code", -15)), raising=False)
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


@pytest.mark.parametrize("payload", [None, {}, {"confirmation": "wrong"}, {"confirmation": "RESET_CH03"}])
def test_eval_start_requires_its_server_side_confirmation(monkeypatch, payload):
    from app.api import jobs

    runner = FakeRunner()
    monkeypatch.setattr(jobs, "get_job_runner", lambda: runner)
    client = TestClient(app)
    response = client.post("/api/jobs/eval-rag", json=payload) if payload is not None else client.post("/api/jobs/eval-rag")
    assert response.status_code == 422
    assert runner.started == []
    assert client.post("/api/jobs/eval-rag", json={"confirmation": "RUN_EVAL_RAG"}).status_code == 202
    assert runner.started == ["eval-rag"]


def test_eval_job_rejects_user_supplied_argv(monkeypatch):
    from app.api import jobs

    runner = FakeRunner()
    monkeypatch.setattr(jobs, "get_job_runner", lambda: runner)
    response = TestClient(app).post("/api/jobs/eval-rag", json={
        "confirmation": "RUN_EVAL_RAG", "argv": ["-c", "print('untrusted')"]})
    assert response.status_code == 422
    assert runner.started == []


@pytest.mark.parametrize("mode,flags", [("resume", []), ("retry", ["--retry-skipped"]), ("fresh", ["--fresh"])])
def test_eval_api_maps_modes_to_fixed_commands(monkeypatch, tmp_path, mode, flags):
    from app.api import jobs
    from app.core.jobs import JobRunner

    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "eval_ch04.py").touch()
    launched = []
    class Process:
        def poll(self): return None
    runner = JobRunner(project_root=tmp_path, popen=lambda argv, **_: (launched.append(argv), Process())[1])
    monkeypatch.setattr(jobs, "get_job_runner", lambda: runner)
    client = TestClient(app)
    assert client.post("/api/jobs/eval-rag", json={"mode": mode}).status_code == 422
    response = client.post("/api/jobs/eval-rag", json={"confirmation": "RUN_EVAL_RAG", "mode": mode})
    assert response.status_code == 202
    assert launched == [[sys.executable, "-m", "scripts.eval_ch04", *flags]]
    assert response.json()["mode"] == mode
    assert client.get("/api/jobs/eval-rag").json()["mode"] == mode


@pytest.mark.parametrize("name,mode", [("eval-rag", "--fresh --data private"), ("kb-preview", "fresh")])
def test_job_modes_cannot_become_arbitrary_arguments(monkeypatch, name, mode):
    from app.api import jobs
    runner = FakeRunner()
    monkeypatch.setattr(jobs, "get_job_runner", lambda: runner)
    response = TestClient(app).post(f"/api/jobs/{name}", json={"confirmation": "RUN_EVAL_RAG", "mode": mode})
    assert response.status_code == 422
    assert runner.started == []


def test_eval_modes_reach_real_checkpoint_retry_and_new_run(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from app.api import jobs
    from app.core.jobs import JobRunner
    from scripts import eval_ch04
    from tests.test_eval_ch04_runner import cases, retrieve

    counts = {"retrieval": 0, "generation": 0, "judge": 0}
    failing = True
    async def retriever(case, strategy):
        counts["retrieval"] += 1
        return await retrieve(case, strategy)
    async def generator(case, result):
        counts["generation"] += 1
        if failing and case["id"] == "A1":
            raise TimeoutError("upstream temporarily unavailable")
        return {"answer": "满99元包邮", "refused": case["id"] == "D1"}
    async def judge(*args):
        counts["judge"] += 1
        return {"faithful": True, "reason": "有依据"}
    async def live(args, rows, resume):
        return await eval_ch04.evaluate(cases(), retriever=retriever, generator=generator,
            judge=judge, resume=resume, retry_skipped=args.retry_skipped, report_path=args.report)
    def popen(argv, **kwargs):
        code = eval_ch04.main(argv[3:])
        return SimpleNamespace(poll=lambda: code)
    monkeypatch.setattr(eval_ch04, "DEFAULT_REPORT", tmp_path / "report.json")
    monkeypatch.setattr(eval_ch04, "_live", live)
    monkeypatch.setattr(jobs, "get_job_runner", lambda: runner)
    runner = JobRunner(log_dir=tmp_path / "logs", popen=popen)
    client = TestClient(app)
    def start(mode):
        counts.update(retrieval=0, generation=0, judge=0)
        response = client.post("/api/jobs/eval-rag", json={"confirmation": "RUN_EVAL_RAG", "mode": mode})
        assert response.status_code == 202
        return response.json()
    assert start("fresh")["exit_code"] == 2
    assert counts == {"retrieval": 8, "generation": 8, "judge": 0}
    failing = False
    assert start("resume")["exit_code"] == 2
    assert counts == {"retrieval": 0, "generation": 0, "judge": 0}
    assert start("retry")["exit_code"] == 0
    assert counts == {"retrieval": 0, "generation": 4, "judge": 1}
    assert start("fresh")["exit_code"] == 0
    assert counts == {"retrieval": 8, "generation": 8, "judge": 1}
    runner.status("eval-rag")


def test_eval_runner_uses_fixed_python_module_and_explicit_availability(tmp_path):
    from app.core.jobs import JobRunner, JobUnavailable

    calls = []
    class Process:
        pid = 123
        def poll(self): return None
    runner = JobRunner(project_root=tmp_path, log_dir=tmp_path / "logs",
                       popen=lambda argv, **kwargs: (calls.append((argv, kwargs)), Process())[1])
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "eval_ch03.py").touch()
    with pytest.raises(JobUnavailable):
        runner.start("eval-rag")
    (scripts / "eval_ch04.py").touch()
    started = runner.start("eval-rag")
    assert started["state"] == "running"
    assert calls[0][0] == [sys.executable, "-m", "scripts.eval_ch04"]
    assert calls[0][1]["shell"] is False


def test_only_one_eval_process_can_start_during_concurrent_requests(tmp_path):
    from app.core.jobs import JobAlreadyRunning, JobRunner

    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "eval_ch04.py").touch()
    entered = Event()
    release = Event()
    calls = []
    class Process:
        pid = 123
        def poll(self): return None
    def popen(argv, **kwargs):
        calls.append(argv)
        entered.set()
        assert release.wait(3)
        return Process()
    runner = JobRunner(project_root=tmp_path, log_dir=tmp_path / "logs", popen=popen)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(runner.start, "eval-rag")
        assert entered.wait(3)
        second = pool.submit(runner.start, "eval-rag")
        release.set()
        assert first.result()["state"] == "running"
        with pytest.raises(JobAlreadyRunning):
            second.result()
    assert len(calls) == 1


def test_eval_status_and_tail_hide_credentials(tmp_path, monkeypatch):
    from app.api import jobs
    from app.core.jobs import JobRunner

    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "eval_ch04.py").touch()
    class Process:
        pid = 123
        code = None
        def poll(self): return self.code
    process = Process()
    runner = JobRunner(project_root=tmp_path, log_dir=tmp_path / "logs",
                       popen=lambda *_args, **_kwargs: process)
    monkeypatch.setattr(jobs, "get_job_runner", lambda: runner)
    client = TestClient(app)
    assert client.post("/api/jobs/eval-rag", json={"confirmation": "RUN_EVAL_RAG"}).status_code == 202
    record = runner._jobs["eval-rag"]
    record["log_file"].write("Authorization: Bearer sensitive-value\nfinished\n")
    record["log_file"].flush()
    status = client.get("/api/jobs/eval-rag", params={"limit": 2}).json()
    assert status["state"] == "running"
    assert "sensitive-value" not in status["log_tail"]
    assert "Bearer [redacted]" in status["log_tail"]
    process.code = 0
    assert client.post("/api/jobs/eval-rag/stop").json()["state"] == "succeeded"


def test_eval_job_is_marked_heavy_in_job_status(tmp_path):
    from app.core.jobs import JobRunner

    runner = JobRunner(project_root=tmp_path, log_dir=tmp_path / "logs")
    assert runner.status("eval-rag")["heavy"] is True
    assert runner.status("kb-reset")["heavy"] is True
    assert runner.status("kb-preview")["heavy"] is False


def test_eval_key_assignments_are_redacted_in_every_job_read_path(tmp_path, monkeypatch):
    from app.api import admin, jobs
    from app.core.jobs import JobRunner

    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "eval_ch04.py").touch()
    class Process:
        pid = 123
        code = None
        def poll(self): return self.code
    process = Process()
    runner = JobRunner(project_root=tmp_path, log_dir=tmp_path / "logs",
                       popen=lambda *_args, **_kwargs: process)
    monkeypatch.setattr(jobs, "get_job_runner", lambda: runner)
    monkeypatch.setattr(admin, "get_job_runner", lambda: runner)
    monkeypatch.setattr(admin, "kb_overview", lambda: None)
    monkeypatch.setattr(admin, "rag_eval_overview", lambda: None)
    client = TestClient(app)
    assert client.post("/api/jobs/eval-rag", json={"confirmation": "RUN_EVAL_RAG"}).status_code == 202
    record = runner._jobs["eval-rag"]
    record["log_file"].write(
        "CHAT_API_KEY=chat-secret EMBED_API_KEY='embed-secret' RERANK_API_KEY=\"rerank-secret\" "
        "JUDGE_API_KEY = judge-secret CUSTOM_API_KEY=custom-secret MESSAGE=ready\n")
    record["log_file"].flush()
    process.code = 0
    outputs = [client.get("/api/jobs/eval-rag").json()["log_tail"],
               next(item for item in client.get("/api/jobs").json() if item["name"] == "eval-rag")["log_tail"],
               client.post("/api/jobs/eval-rag/stop").json()["log_tail"],
               next(item for item in client.get("/api/admin/overview").json()["jobs"]
                    if item["name"] == "eval-rag")["log_tail"]]
    for output in outputs:
        assert all(secret not in output for secret in
                   ("chat-secret", "embed-secret", "rerank-secret", "judge-secret", "custom-secret"))
        assert output.count("[redacted]") == 5
        assert "MESSAGE=ready" in output


def test_stop_waits_for_slow_eval_spawn_then_stops_the_owned_process(tmp_path, monkeypatch):
    from app.core import jobs
    from app.core.jobs import JobRunner

    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "eval_ch04.py").touch()
    spawn_entered = Event()
    allow_spawn = Event()
    stop_entered = Event()
    class Process:
        pid = 456
        code = None
        def poll(self): return self.code
    process = Process()
    def popen(*_args, **_kwargs):
        spawn_entered.set()
        assert allow_spawn.wait(3)
        return process
    runner = JobRunner(project_root=tmp_path, log_dir=tmp_path / "logs", popen=popen)
    monkeypatch.setattr(jobs.os, "name", "posix")
    monkeypatch.setattr(jobs.os, "killpg", lambda _pid, _signal: setattr(process, "code", -15), raising=False)
    def stop():
        stop_entered.set()
        return runner.stop("eval-rag")
    with ThreadPoolExecutor(max_workers=2) as pool:
        started = pool.submit(runner.start, "eval-rag")
        assert spawn_entered.wait(3)
        stopped = pool.submit(stop)
        assert stop_entered.wait(3)
        try:
            with pytest.raises(FutureTimeoutError):
                stopped.result(timeout=0.2)
        finally:
            allow_spawn.set()
        assert started.result()["state"] == "running"
        assert stopped.result()["state"] == "stopped"
    assert runner.status("eval-rag")["state"] == "stopped"
