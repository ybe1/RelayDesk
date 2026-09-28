import json

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.mark.parametrize("content,status", [(None, "missing"), ("{", "corrupt"), ("[]", "corrupt"),
    ('{"mrr":NaN}', "corrupt"), ('{"mrr":Infinity}', "corrupt"),
    ('{"generation":{"elapsed":1e400}}', "corrupt"),
    ('{"cases":[{"timings":[-1e400]}]}', "corrupt"),
    ('{"retrieval":{"vector":{"overall":{"mrr":1e400}}}}', "corrupt")])
def test_unavailable_report_is_usable(monkeypatch, tmp_path, content, status):
    from app.api import rageval
    path = tmp_path / "report.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(rageval, "REPORT_PATH", path)
    response = TestClient(app).get("/api/rag-eval/overview")
    assert response.status_code == 200
    assert response.json() == {"report_status": status, "report": None, "best_mrr": None}


def test_partial_report_forwards_all_metrics_without_recalculation(monkeypatch, tmp_path):
    from app.api import rageval
    report = {"meta": {"retrieval_status": "complete"}, "generation": {
        "status": "partial", "faithfulness": {"faithfulness_rate": None, "completed": 0}},
        "retrieval": {"vector": {"overall": {"mrr": 0.8}}, "bm25": {"overall": {"mrr": 0.9}},
                      "hybrid": {"overall": {"mrr": None}}},
        "cases": {"vector": [{"metrics": {"mrr": 0.1}}]}, "best": {"strategy": "vector", "mrr": 0.8}}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(rageval, "REPORT_PATH", path)
    result = TestClient(app).get("/api/rag-eval/overview").json()
    assert result["report"] == report
    assert result["report_status"] == "partial"
    assert result["best_mrr"] == {"strategy": "bm25", "mrr": 0.9}


def test_report_without_metrics_has_no_invented_best(monkeypatch, tmp_path):
    from app.api import rageval
    path = tmp_path / "report.json"
    path.write_text('{"generation":{"status":"partial"}}', encoding="utf-8")
    monkeypatch.setattr(rageval, "REPORT_PATH", path)
    result = TestClient(app).get("/api/rag-eval/overview").json()
    assert result["best_mrr"] is None
