from datetime import datetime

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import KnowledgeChunk, LowConfidenceQuestion, ReviewQueue
from app.main import app


def _sessions():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with engine.begin() as connection:
        connection.exec_driver_sql("""CREATE TABLE review_queue (
          id INTEGER PRIMARY KEY AUTOINCREMENT, normalized_question VARCHAR(512) NOT NULL,
          ai_suggested_answer TEXT, occurrence_count INTEGER NOT NULL DEFAULT 1,
          review_status VARCHAR(64) NOT NULL DEFAULT '待审', approved_answer TEXT,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
        connection.exec_driver_sql("""CREATE TABLE knowledge_chunks (
          id INTEGER PRIMARY KEY AUTOINCREMENT, category VARCHAR(255) NOT NULL, questions TEXT NOT NULL,
          answer TEXT NOT NULL, section_path VARCHAR(512), content_type VARCHAR(32), is_key_clause INTEGER DEFAULT 0,
          prev_chunk_id INTEGER, next_chunk_id INTEGER, vector_id VARCHAR(64), vectorize_status VARCHAR(64) DEFAULT 'pending',
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
        connection.exec_driver_sql("""CREATE TABLE low_confidence_questions (
          id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id INTEGER, raw_question TEXT NOT NULL,
          source VARCHAR(64) NOT NULL, reason TEXT, retrieved_chunks JSON, matched_review_id INTEGER,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    return sessionmaker(engine, expire_on_commit=False)


def _seed(factory):
    with factory() as session:
        review = ReviewQueue(normalized_question="羊毛衫怎么洗？", ai_suggested_answer="建议干洗",
                             occurrence_count=3, review_status="待审", created_at=datetime.now(), updated_at=datetime.now())
        session.add(review); session.flush()
        session.add(LowConfidenceQuestion(conversation_id=None, raw_question="毛衣咋洗",
            source="user_feedback", reason="没解决", retrieved_chunks=[{"id": 8, "score": 0.2}],
            matched_review_id=review.id, created_at=datetime(2026, 1, 1)))
        session.commit()
        return review.id


def test_list_detail_and_approve_are_transactional_and_idempotent():
    factory = _sessions(); review_id = _seed(factory)
    app.state.review_session_factory = factory
    client = TestClient(app)

    listing = client.get("/api/review-queue").json()
    assert listing["items"][0]["id"] == review_id
    assert listing["items"][0]["occurrence_count"] == 3
    detail = client.get(f"/api/review-queue/{review_id}").json()
    assert detail["raw_questions"][0]["raw_question"] == "毛衣咋洗"
    assert detail["raw_questions"][0]["retrieved_chunks"][0]["id"] == 8

    payload = {"decision": "通过", "approved_answer": "按洗标清洗", "expected_status": "待审"}
    response = client.post(f"/api/review-queue/{review_id}/decision", json=payload)
    assert response.status_code == 200
    assert response.json()["vectorize_status"] == "pending"
    assert client.post(f"/api/review-queue/{review_id}/decision", json=payload).status_code == 200
    with factory() as session:
        chunks = session.query(KnowledgeChunk).all()
        assert len(chunks) == 1
        assert (chunks[0].category, chunks[0].content_type, chunks[0].section_path) == (
            "用户问题补充", "faq", f"数据飞轮 / review_queue/{review_id}")
    del app.state.review_session_factory


def test_changed_answer_and_late_reject_return_conflict():
    factory = _sessions(); review_id = _seed(factory)
    app.state.review_session_factory = factory
    client = TestClient(app)
    first = {"decision": "通过", "approved_answer": "按洗标清洗", "expected_status": "待审"}
    assert client.post(f"/api/review-queue/{review_id}/decision", json=first).status_code == 200
    changed = {**first, "approved_answer": "换一个答案"}
    assert client.post(f"/api/review-queue/{review_id}/decision", json=changed).status_code == 409
    reject = {"decision": "驳回", "expected_status": "待审"}
    assert client.post(f"/api/review-queue/{review_id}/decision", json=reject).status_code == 409
    del app.state.review_session_factory


def test_approval_rejects_utf8_fields_over_collection_caps():
    factory = _sessions(); review_id = _seed(factory)
    app.state.review_session_factory = factory
    response = TestClient(app).post(f"/api/review-queue/{review_id}/decision", json={
        "decision": "通过", "approved_answer": "猫" * 20_000, "expected_status": "待审"})
    assert response.status_code == 422
    del app.state.review_session_factory


def test_client_cannot_supply_terminal_expected_status():
    factory=_sessions(); review_id=_seed(factory); app.state.review_session_factory=factory
    response=TestClient(app).post(f"/api/review-queue/{review_id}/decision",json={
        "decision":"驳回","expected_status":"通过"})
    assert response.status_code == 422
    del app.state.review_session_factory
