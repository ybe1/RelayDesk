from fastapi.testclient import TestClient
import json

from app.main import app


def test_admin_overview_degrades_when_one_aggregate_dependency_is_down(monkeypatch):
    from app.api import admin
    monkeypatch.setattr(admin, "kb_overview", lambda: {"total": 3, "consistent": None})
    monkeypatch.setattr(admin, "job_summary", lambda: (_ for _ in ()).throw(RuntimeError("runner down")))
    response = TestClient(app).get("/api/admin/overview")
    assert response.status_code == 200
    assert response.json()["knowledge"]["total"] == 3
    assert response.json()["jobs"] is None


def test_admin_eval_card_reuses_the_saved_report_without_writing(monkeypatch, tmp_path):
    from app.api import admin, rageval

    report = {"meta": {"retrieval_status": "complete"},
              "retrieval": {"hybrid": {"overall": {"mrr": 0.75}}},
              "generation": {"status": "partial", "faithfulness": {"faithfulness_rate": None}}}
    path = tmp_path / "rag_eval.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    before = path.read_bytes(), path.stat().st_mtime_ns
    monkeypatch.setattr(rageval, "REPORT_PATH", path)
    monkeypatch.setattr(admin, "kb_overview", lambda: {"total": 1})
    monkeypatch.setattr(admin, "job_summary", lambda: [])
    client = TestClient(app)
    eval_overview = client.get("/api/rag-eval/overview").json()
    admin_overview = client.get("/api/admin/overview").json()
    assert admin_overview["rag_eval"] == eval_overview
    assert admin_overview["rag_eval"]["report"] == report
    assert before == (path.read_bytes(), path.stat().st_mtime_ns)


def test_rag_eval_page_route_and_admin_card_are_available():
    client = TestClient(app)
    page = client.get("/rag-eval")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "rag-eval" in page.text
    admin = client.get("/admin")
    assert admin.status_code == 200
    assert 'id="a-rag-eval"' in admin.text
