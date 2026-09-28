"""Local-only runner for fixed maintenance and evaluation commands."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
JOBS = {
    "kb-preview": [sys.executable, "-m", "scripts.kb", "preview"],
    "kb-build": [sys.executable, "-m", "scripts.kb", "build"],
    "kb-mine": [sys.executable, "-m", "scripts.kb", "mine"],
    "kb-vectorize": [sys.executable, "-m", "scripts.kb", "vectorize"],
    "seed-conv": [sys.executable, "-m", "scripts.kb", "seed-conv"],
    "kb-reset": [sys.executable, "-m", "scripts.kb", "reset", "--confirmed"],
    "eval-retrieval": [sys.executable, "-m", "scripts.eval_ch03", "retrieval"],
    "eval-mining": [sys.executable, "-m", "scripts.eval_ch03", "mining"],
    "eval-rag": [sys.executable, "-m", "scripts.eval_ch04"],
}

# Availability is tied to a named module, never inferred from a job-name prefix.
# Existing KB commands retain their current availability behavior.
JOB_MODULES = {
    "kb-preview": None,
    "kb-build": None,
    "kb-mine": None,
    "kb-vectorize": None,
    "seed-conv": None,
    "kb-reset": None,
    "eval-retrieval": Path("scripts/eval_ch03.py"),
    "eval-mining": Path("scripts/eval_ch03.py"),
    "eval-rag": Path("scripts/eval_ch04.py"),
}
HEAVY_JOBS = frozenset({"kb-reset", "eval-rag"})


class UnknownJob(LookupError):
    pass


class JobAlreadyRunning(RuntimeError):
    pass


class JobUnavailable(RuntimeError):
    pass


class JobRunner:
    """Own only processes started by this runner; never accepts user argv."""

    def __init__(self, project_root: Path = PROJECT_ROOT, log_dir: Path | None = None, popen: Callable = subprocess.Popen):
        self.project_root = project_root
        self.log_dir = log_dir or project_root / "log" / "acceptance"
        self.popen = popen
        self._jobs: dict[str, dict] = {}
        self._start_lock = threading.Lock()

    def _check_name(self, name: str) -> None:
        if name not in JOBS:
            raise UnknownJob(name)

    def _available(self, name: str) -> bool:
        module = JOB_MODULES[name]
        return module is None or (self.project_root / module).is_file()

    def _tail(self, path: Path, limit: int = 200) -> str:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except FileNotFoundError:
            return ""
        # Credentials must never be replayed to the UI, including common JSON,
        # header, and standalone bearer-token forms.
        tail = "\n".join(lines[-max(1, min(limit, 200)):])
        tail = re.sub(r'(?i)("(?:(?:[A-Za-z0-9_]+_)?api[_-]?key|token|secret|password|authorization)"\s*:\s*")[^"]*(")', r"\1[redacted]\2", tail)
        tail = re.sub(r"(?i)(\b(?:(?:[A-Za-z0-9_]+_)?api[_-]?key|token|secret|password)\s*[:=]\s*[\"'])[^\"']*([\"'])", r"\1[redacted]\2", tail)
        tail = re.sub(r"(?i)(\bauthorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;\"}]+", r"\1Bearer [redacted]", tail)
        tail = re.sub(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]+=*", "Bearer [redacted]", tail)
        return re.sub(r"(?i)\b((?:[A-Za-z0-9_]+_)?api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,;\"}]+", r"\1=[redacted]", tail)

    def tail(self, name: str, limit: int = 200) -> str:
        self._check_name(name)
        if not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200")
        record = self._jobs.get(name)
        return "" if record is None else self._tail(record["log_path"], limit)

    def _state(self, record: dict) -> tuple[str, int | None]:
        code = record["process"].poll()
        if code is None:
            return "running", None
        if record.get("stop_requested"):
            return "stopped", code
        return ("succeeded" if code == 0 else "failed"), code

    def _payload(self, record: dict, limit: int = 200) -> dict:
        state, code = self._state(record)
        return {"job_id": record["job_id"], "name": record["name"], "state": state,
                "exit_code": code, "log_tail": self._tail(record["log_path"], limit),
                "heavy": record["name"] in HEAVY_JOBS}

    def start(self, name: str) -> dict:
        self._check_name(name)
        with self._start_lock:
            if not self._available(name):
                raise JobUnavailable(f"{name} is unavailable: required module {JOB_MODULES[name]} is missing")
            if name in self._jobs and self._jobs[name]["process"].poll() is None:
                raise JobAlreadyRunning(name)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            path = self.log_dir / f"{name}-{int(time.time() * 1000)}.log"
            log_file = path.open("w", encoding="utf-8")
            kwargs = {"cwd": self.project_root, "stdout": log_file, "stderr": subprocess.STDOUT, "shell": False}
            if os.name == "nt":
                kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            else:
                kwargs["start_new_session"] = True
            try:
                process = self.popen(JOBS[name], **kwargs)
            except Exception:
                log_file.close()
                raise
            self._jobs[name] = {"job_id": uuid.uuid4().hex, "name": name, "process": process,
                                "log_path": path, "log_file": log_file, "stop_requested": False}
            return self._payload(self._jobs[name])

    def status(self, name: str, limit: int = 200) -> dict:
        self._check_name(name)
        record = self._jobs.get(name)
        if record is None:
            return {"job_id": None, "name": name, "state": "stopped", "exit_code": None,
                    "log_tail": "", "heavy": name in HEAVY_JOBS}
        result = self._payload(record, limit)
        if result["state"] != "running" and not record["log_file"].closed:
            record["log_file"].close()
        return result

    def list(self) -> list[dict]:
        return [self.status(name) for name in JOBS]

    def stop(self, name: str) -> dict:
        self._check_name(name)
        # A stop arriving during Popen must wait until start owns the process.
        with self._start_lock:
            record = self._jobs.get(name)
            if record is None:
                return self.status(name)
            process = record["process"]
            # A terminal process keeps its actual succeeded/failed state. A
            # failed or asynchronous termination still owns a live process, so a
            # subsequent start remains blocked until poll() proves it exited.
            if process.poll() is not None:
                return self.status(name)
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False, shell=False,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    os.killpg(process.pid, 15)
            except OSError:
                return self.status(name)
            record["stop_requested"] = True
            return self.status(name)

    def shutdown(self) -> None:
        for name in tuple(self._jobs):
            if self._state(self._jobs[name])[0] == "running":
                self.stop(name)
