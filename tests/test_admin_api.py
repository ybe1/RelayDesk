from fastapi.testclient import TestClient

from app.main import app


def test_admin_overview_degrades_when_one_aggregate_dependency_is_down(monkeypatch):
    from app.api import admin
    monkeypatch.setattr(admin, "kb_overview", lambda: {"total": 3, "consistent": None})
    monkeypatch.setattr(admin, "job_summary", lambda: (_ for _ in ()).throw(RuntimeError("runner down")))
    response = TestClient(app).get("/api/admin/overview")
    assert response.status_code == 200
    assert response.json()["knowledge"]["total"] == 3
    assert response.json()["jobs"] is None
