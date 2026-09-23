from fastapi.testclient import TestClient

from app.main import app


def test_management_static_pages_are_reachable():
    client = TestClient(app)
    for path in ("/", "/kb", "/admin", "/static/admin.css", "/static/kb.js", "/static/admin.js"):
        response = client.get(path)
        assert response.status_code == 200
