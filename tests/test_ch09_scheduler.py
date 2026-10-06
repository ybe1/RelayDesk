import json
from datetime import datetime
from zoneinfo import ZoneInfo

from app.evaluation.scheduler import FlywheelScheduler, MaintenanceLoop


SHANGHAI = ZoneInfo("Asia/Shanghai")


class Runner:
    def __init__(self): self.started = []
    def start(self, name, *, mode=None): self.started.append((name,mode)); return {"job_id": "job-1"}


class Lock:
    def __init__(self, allowed=True): self.allowed = allowed
    def __enter__(self): return self.allowed
    def __exit__(self, *_): pass


def test_first_enable_only_schedules_next_slot(tmp_path):
    runner = Runner()
    scheduler = FlywheelScheduler(runner, tmp_path / "schedule.json", lock_factory=Lock)
    decision = scheduler.tick(datetime(2026, 10, 6, 9, tzinfo=SHANGHAI))
    assert decision.action == "enabled"
    assert runner.started == []


def test_startup_catches_up_only_latest_missed_slot_and_never_retries_paid_attempt(tmp_path):
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps({"enabled_at": "2026-10-01T00:00:00+08:00",
        "attempted_slot": "2026-10-12T02:00:00+08:00"}), encoding="utf-8")
    runner = Runner(); scheduler = FlywheelScheduler(runner, path, lock_factory=Lock)
    now = datetime(2026, 10, 19, 9, tzinfo=SHANGHAI)
    decision = scheduler.tick(now)
    assert decision.slot == "2026-10-19T02:00:00+08:00"
    assert decision.run_count == 1
    assert runner.started == [("eval-flywheel","scheduled")]
    assert scheduler.tick(now).run_count == 0
    assert runner.started == [("eval-flywheel","scheduled")]


def test_multi_worker_lock_prevents_duplicate_start(tmp_path):
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps({"enabled_at": "2026-10-01T00:00:00+08:00",
        "attempted_slot": "2026-10-12T02:00:00+08:00"}), encoding="utf-8")
    runner = Runner()
    decision = FlywheelScheduler(runner, path, lock_factory=lambda: Lock(False)).tick(
        datetime(2026, 10, 19, 9, tzinfo=SHANGHAI))
    assert decision.action == "locked"
    assert runner.started == []


def test_maintenance_submits_processing_and_index_recovery():
    runner=Runner(); MaintenanceLoop(runner).tick()
    assert runner.started == [("flywheel-process",None),("flywheel-index-recover",None)]


def test_complete_report_repairs_missing_sql_row_without_paid_rerun(tmp_path):
    path=tmp_path/"schedule.json"; reports=tmp_path/"reports"; reports.mkdir()
    slot="2026-10-19T02:00:00+08:00"; run_id=f"scheduled:{slot}"
    path.write_text(json.dumps({"attempted_slot":slot,"status":"manual_recovery_required","run_id":run_id}),encoding="utf-8")
    (reports/f"{run_id.replace(':','_')}.json").write_text(json.dumps({
        "meta":{"retrieval_status":"complete"},"generation":{"status":"complete"}}),encoding="utf-8")
    runner=Runner(); decision=FlywheelScheduler(runner,path,lock_factory=Lock,report_dir=reports).tick(
        datetime(2026,10,19,9,tzinfo=SHANGHAI))
    assert decision.action=="repair_started"
    assert runner.started==[("eval-flywheel","scheduled")]

