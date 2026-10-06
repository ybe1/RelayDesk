from __future__ import annotations

import json
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import text


LOGGER = logging.getLogger(__name__)
SHANGHAI = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class SchedulerDecision:
    action: str
    slot: str | None = None
    run_count: int = 0


def _latest_slot(now: datetime) -> datetime:
    local = now.astimezone(SHANGHAI)
    monday = (local - timedelta(days=local.weekday())).date()
    slot = datetime.combine(monday, time(2), SHANGHAI)
    return slot if slot <= local else slot - timedelta(days=7)


def mysql_scheduler_lock(session_factory):
    @contextmanager
    def acquire():
        with session_factory() as session:
            acquired = session.execute(text("SELECT GET_LOCK('mewhelp_ch09_eval_schedule', 0)")).scalar() == 1
            try:
                yield acquired
            finally:
                if acquired:
                    session.execute(text("SELECT RELEASE_LOCK('mewhelp_ch09_eval_schedule')"))
    return acquire


class FlywheelScheduler:
    def __init__(self, runner, state_path: Path, *, lock_factory, report_dir: Path | None = None):
        self.runner = runner
        self.state_path = Path(state_path)
        self.lock_factory = lock_factory
        self.report_dir=Path(report_dir) if report_dir else self.state_path.parents[2]/"reports"

    def _complete_report_exists(self,state):
        run_id=str(state.get("run_id", ""))
        if not run_id:return False
        path=self.report_dir/f"{run_id.replace(':','_')}.json"
        try:
            report=json.loads(path.read_text(encoding="utf-8"))
            return report.get("meta",{}).get("retrieval_status")=="complete" and report.get("generation",{}).get("status")=="complete"
        except (OSError,json.JSONDecodeError): return False

    def _read(self) -> dict | None:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None

    def _write(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_name(self.state_path.name + ".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        temporary.replace(self.state_path)

    def tick(self, now: datetime) -> SchedulerDecision:
        if now.tzinfo is None:
            raise ValueError("scheduler clock must be timezone-aware")
        with self.lock_factory() as acquired:
            if not acquired:
                return SchedulerDecision("locked")
            slot = _latest_slot(now).isoformat()
            state = self._read()
            if state is None:
                self._write({"enabled_at": now.astimezone(SHANGHAI).isoformat(), "attempted_slot": slot,
                             "status": "enabled"})
                return SchedulerDecision("enabled", slot)
            if state.get("attempted_slot") == slot:
                if state.get("status")=="manual_recovery_required" and self._complete_report_exists(state):
                    try:
                        job=self.runner.start("eval-flywheel",mode="scheduled")
                        state.update(status="started",job_id=job.get("job_id")); self._write(state)
                        return SchedulerDecision("repair_started",slot,0)
                    except Exception:
                        return SchedulerDecision("repair_waiting",slot,0)
                return SchedulerDecision("already_attempted", slot)
            # Persist the paid-run attempt before process creation. A crash or
            # failed call must be resumed manually and never charged again at startup.
            state.update(attempted_slot=slot, status="starting", run_id=f"scheduled:{slot}")
            self._write(state)
            try:
                job = self.runner.start("eval-flywheel", mode="scheduled")
                state.update(status="started", job_id=job.get("job_id"))
                self._write(state)
                return SchedulerDecision("started", slot, 1)
            except Exception as error:
                state.update(status="manual_recovery_required", error=type(error).__name__)
                self._write(state)
                LOGGER.warning("scheduled evaluation requires manual recovery: %s", type(error).__name__)
                return SchedulerDecision("manual_recovery_required", slot)


class SchedulerLoop:
    def __init__(self, scheduler: FlywheelScheduler, interval_seconds: float = 60):
        self.scheduler, self.interval_seconds = scheduler, interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="ch09-eval-scheduler")
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                self.scheduler.tick(datetime.now(SHANGHAI))
            except Exception:
                LOGGER.exception("evaluation scheduler tick failed")
            self._stop.wait(self.interval_seconds)

    def close(self, timeout: float = 2):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)


class MaintenanceLoop:
    """Submit fixed idempotent flywheel jobs; JobRunner prevents overlap."""
    def __init__(self, runner, interval_seconds: float = 60):
        self.runner,self.interval_seconds=runner,interval_seconds
        self._stop=threading.Event(); self._thread=None
    def tick(self):
        for name in ("flywheel-process","flywheel-index-recover"):
            try: self.runner.start(name)
            except Exception as error:
                if type(error).__name__ != "JobAlreadyRunning":
                    LOGGER.warning("maintenance job %s not started: %s",name,type(error).__name__)
    def start(self):
        if self._thread is not None:return
        self._thread=threading.Thread(target=self._run,daemon=True,name="ch09-maintenance")
        self._thread.start()
    def _run(self):
        while not self._stop.is_set():
            self.tick(); self._stop.wait(self.interval_seconds)
    def close(self,timeout:float=2):
        self._stop.set()
        if self._thread is not None:self._thread.join(timeout)

