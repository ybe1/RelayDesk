"""Local-only runner for the fixed Ch03 maintenance commands."""

from __future__ import annotations

import os
import re
import subprocess
import sys
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
}


class UnknownJob(LookupError):
    pass


class JobAlreadyRunning(RuntimeError):
    pass


class JobRunner:
    """Own only processes started by this runner; never accepts user argv."""

    def __init__(self, project_root: Path = PROJECT_ROOT, log_dir: Path | None = None, popen: Callable = subprocess.Popen):
        self.project_root = project_root
        self.log_dir = log_dir or project_root / "log" / "acceptance"
        self.popen = popen
        self._jobs: dict[str, dict] = {}

    def _check_name(self, name: str) -> None:
        if name not in JOBS:
            raise UnknownJob(name)

    def _tail(self, path: Path, limit: int = 200) -> str:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except FileNotFoundError:
            return ""
        # Environment variables and bearer-like values must never be replayed to the UI.
        tail = "\n".join(lines[-max(1, min(limit, 200)):])
        return re.sub(r"(?i)\b(api[_-]?key|token|authorization)\s*[:=]\s*\S+", r"\1=[redacted]", tail)

    def _state(self, record: dict) -> tuple[str, int | None]:
        code = record["process"].poll()
        if record.get("stopped"):
            return "stopped", code
        if code is None:
            return "running", None
        return ("succeeded" if code == 0 else "failed"), code

    def _payload(self, record: dict, limit: int = 200) -> dict:
        state, code = self._state(record)
        return {"job_id": record["job_id"], "name": record["name"], "state": state,
                "exit_code": code, "log_tail": self._tail(record["log_path"], limit)}

    def start(self, name: str) -> dict:
        self._check_name(name)
        if name in self._jobs and self._state(self._jobs[name])[0] == "running":
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
                            "log_path": path, "log_file": log_file, "stopped": False}
        return self._payload(self._jobs[name])

    def status(self, name: str, limit: int = 200) -> dict:
        self._check_name(name)
        record = self._jobs.get(name)
        if record is None:
            return {"job_id": None, "name": name, "state": "stopped", "exit_code": None, "log_tail": ""}
        result = self._payload(record, limit)
        if result["state"] != "running" and not record["log_file"].closed:
            record["log_file"].close()
        return result

    def list(self) -> list[dict]:
        return [self.status(name) for name in JOBS]

    def stop(self, name: str) -> dict:
        self._check_name(name)
        record = self._jobs.get(name)
        if record is None:
            return self.status(name)
        process = record["process"]
        if process.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False, shell=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                os.killpg(process.pid, 15)
        record["stopped"] = True
        return self.status(name)

    def shutdown(self) -> None:
        for name in tuple(self._jobs):
            if self._state(self._jobs[name])[0] == "running":
                self.stop(name)
