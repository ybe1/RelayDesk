from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.api.feedback import router
from app.db.models import LowConfidenceQuestion


class Graph:
    checkpointer = object()
    def __init__(self, messages): self.messages = messages
    async def aget_state(self, _config): return SimpleNamespace(values={"messages": self.messages})


class Service:
    def __init__(self, messages): self.graph = Graph(messages)
    @staticmethod
    def _config(cid): return {"configurable": {"thread_id": f"conversation:{cid}"}}


def client(tmp_path, *, snapshot=True):
    engine = create_engine(f"sqlite:///{tmp_path / 'feedback.db'}", connect_args={"check_same_thread": False})
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE conversations (id INTEGER PRIMARY KEY, user_id VARCHAR(64), status VARCHAR(16), summary TEXT, summary_upto_msg_id INTEGER, layer1_from_msg_id INTEGER, created_at DATETIME, updated_at DATETIME)")
        connection.exec_driver_sql("CREATE TABLE messages (id INTEGER PRIMARY KEY, conversation_id INTEGER, role VARCHAR(16), content TEXT, tool_calls JSON, tool_call_id VARCHAR(64), message_key VARCHAR(128), created_at DATETIME)")
        connection.exec_driver_sql("CREATE TABLE review_queue (id INTEGER PRIMARY KEY)")
        connection.exec_driver_sql("CREATE TABLE low_confidence_questions (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id INTEGER, raw_question TEXT NOT NULL, source VARCHAR(32) NOT NULL, reason TEXT, retrieved_chunks JSON, matched_review_id INTEGER, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)")
        connection.exec_driver_sql("INSERT INTO conversations (id,user_id,status) VALUES (1,'u1','进行中')")
        connection.exec_driver_sql("INSERT INTO messages (id,conversation_id,role,content,message_key) VALUES (10,1,'user','第一问','turn-a:user'),(11,1,'assistant','第一答','turn-a:final'),(20,1,'user','第二问','turn-b:user'),(21,1,'assistant','第二答','turn-b:final')")
    sessions = sessionmaker(engine, expire_on_commit=False)
    kwargs = {"evidence_snapshot": {"hits": [{"chunk_id": 7, "answer": "第一轮证据", "rerank_score": .8}]}} if snapshot else {"evidence_snapshot": None}
    app = FastAPI(); app.include_router(router)
    app.state.feedback_session_factory = sessions
    app.state.workflow_service = Service([AIMessage(content="第一答", id="turn-a:final", additional_kwargs=kwargs),
        AIMessage(content="第二答", id="turn-b:final", additional_kwargs={"evidence_snapshot": None})])
    return TestClient(app), sessions


def payload(message_id=11, user="u1"):
    return {"user_id": user, "conversation_id": 1, "assistant_message_id": message_id, "helpful": False}


def test_feedback_uses_selected_turn_snapshot_and_is_idempotent(tmp_path):
    http, sessions = client(tmp_path)
    first = http.post("/api/chat/feedback", json=payload())
    second = http.post("/api/chat/feedback", json=payload())
    assert first.status_code == second.status_code == 200
    assert first.json()["low_confidence_id"] == second.json()["low_confidence_id"]
    with sessions() as session:
        rows = list(session.scalars(select(LowConfidenceQuestion)))
        assert len(rows) == 1 and rows[0].raw_question == "第一问"
        assert rows[0].retrieved_chunks[0]["chunk_id"] == 7


def test_feedback_without_retrieval_keeps_sql_null(tmp_path):
    http, sessions = client(tmp_path)
    assert http.post("/api/chat/feedback", json=payload(21)).status_code == 200
    with sessions() as session:
        assert session.scalar(select(LowConfidenceQuestion)).retrieved_chunks is None


def test_feedback_checks_owner_and_assistant_role(tmp_path):
    http, _ = client(tmp_path)
    assert http.post("/api/chat/feedback", json=payload(user="other")).status_code == 403
    assert http.post("/api/chat/feedback", json=payload(10)).status_code == 422
