from __future__ import annotations

from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db.models import Conversation, KnowledgeChunk, Message, QAExtractionStaging


@pytest.fixture
def session_factory():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as connection:
        connection.exec_driver_sql("""
            CREATE TABLE conversations (
                id INTEGER PRIMARY KEY, user_id VARCHAR(64) NOT NULL,
                status VARCHAR NOT NULL DEFAULT '进行中',
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.exec_driver_sql("""
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL,
                role VARCHAR NOT NULL, content TEXT, tool_calls JSON,
                tool_call_id VARCHAR(64), created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.exec_driver_sql("""
            CREATE TABLE qa_extraction_staging (
                id INTEGER PRIMARY KEY, batch_no VARCHAR(64) NOT NULL,
                source_ref VARCHAR(255), question TEXT NOT NULL, answer TEXT NOT NULL,
                status VARCHAR NOT NULL DEFAULT 'extracted',
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.exec_driver_sql("""
            CREATE TABLE knowledge_chunks (
                id INTEGER PRIMARY KEY, category VARCHAR(255) NOT NULL,
                questions TEXT NOT NULL, answer TEXT NOT NULL, section_path VARCHAR(512),
                content_type VARCHAR(32), is_key_clause INTEGER NOT NULL DEFAULT 0,
                prev_chunk_id INTEGER, next_chunk_id INTEGER, vector_id VARCHAR(64),
                vectorize_status VARCHAR NOT NULL DEFAULT 'pending',
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory.begin() as session:
        session.add_all([
            Conversation(id=1, user_id="one"),
            Conversation(id=2, user_id="two"),
            Message(conversation_id=1, role="user", content="退货期限是多久？"),
            Message(conversation_id=1, role="assistant", content="支持七天无理由退货。"),
            Message(conversation_id=1, role="assistant", tool_calls=[{"name": "query_faq"}]),
            Message(conversation_id=1, role="tool", content="内部工具结果：订单 A100 已签收"),
            Message(conversation_id=2, role="user", content="运费如何计算？"),
            Message(conversation_id=2, role="assistant", content="订单 B200 的物流单号是 SF123。"),
        ])
    return factory


class FakeStructuredModel:
    def __init__(self, results):
        self.results = list(results)
        self.inputs = []
        self.schemas = []

    def with_structured_output(self, schema, *, method):
        self.schemas.append((schema, method))
        return self

    def invoke(self, value):
        self.inputs.append(value)
        return self.results.pop(0)


def _staging(session_factory):
    with session_factory() as session:
        return session.scalars(select(QAExtractionStaging).order_by(QAExtractionStaging.id)).all()


def test_extraction_sends_only_user_and_textual_assistant_messages(session_factory):
    from app.knowledge.extraction import ExtractedQA, QASet, extract_batches

    model = FakeStructuredModel([QASet(items=[ExtractedQA(
        conversation_id=1, question="退货期限是多久？", answer="支持七天无理由退货。"
    )])])

    result = extract_batches(session_factory, model, batch_size=30)

    assert result == {"batches": 1, "extracted": 1, "skipped": 0}
    assert model.schemas == [(QASet, "function_calling")]
    transcript = "\n".join(message.content for message in model.inputs[0])
    assert "退货期限是多久？" in transcript
    assert "支持七天无理由退货。" in transcript
    assert "内部工具结果" not in transcript
    assert "query_faq" not in transcript
    assert "订单 B200" not in transcript


def test_extraction_excludes_case_specific_logistics_state_without_an_order_number(session_factory):
    from app.knowledge.extraction import QASet, extract_batches

    with session_factory.begin() as session:
        session.add_all([
            Conversation(id=3, user_id="three"),
            Message(conversation_id=3, role="user", content="我的订单到哪儿了？"),
            Message(conversation_id=3, role="assistant", content="模拟订单的物流显示已揽收。"),
        ])
    model = FakeStructuredModel([QASet(items=[])])

    extract_batches(session_factory, model)

    transcript = "\n".join(message.content for message in model.inputs[0])
    assert "模拟订单的物流显示已揽收。" not in transcript


def test_extraction_redacts_phone_email_name_and_address_before_model_input(session_factory):
    from app.knowledge.extraction import QASet, extract_batches

    with session_factory.begin() as session:
        session.add_all([
            Conversation(id=3, user_id="three"),
            Message(conversation_id=3, role="user", content="姓名：张三，手机号 13800138000，邮箱 zhang@example.com，地址北京市朝阳区望京路 1 号，怎么退货？"),
            Message(conversation_id=3, role="assistant", content="请在订单页申请退货，客服会联系您。"),
        ])
    model = FakeStructuredModel([QASet(items=[])])

    extract_batches(session_factory, model)

    transcript = "\n".join(message.content for message in model.inputs[0])
    assert "13800138000" not in transcript
    assert "zhang@example.com" not in transcript
    assert "张三" not in transcript
    assert "北京市朝阳区望京路 1 号" not in transcript
    assert "怎么退货？" in transcript


@pytest.mark.parametrize("topic", ["退货政策", "运费规则", "售后流程"])
def test_extraction_preserves_unlabeled_topic_at_start_of_message(topic, session_factory):
    from app.knowledge.extraction import QASet, extract_batches

    with session_factory.begin() as session:
        session.add_all([
            Conversation(id=3, user_id="three"),
            Message(conversation_id=3, role="user", content=f"{topic}，请说明一下。"),
            Message(conversation_id=3, role="assistant", content="这是通用说明。"),
        ])
    model = FakeStructuredModel([QASet(items=[])])

    extract_batches(session_factory, model)

    transcript = "\n".join(message.content for message in model.inputs[0])
    assert topic in transcript


def test_extraction_uses_only_the_final_eligible_assistant_answer_per_conversation(session_factory):
    from app.knowledge.extraction import QASet, extract_batches

    with session_factory.begin() as session:
        session.add_all([
            Conversation(id=3, user_id="three"),
            Message(conversation_id=3, role="user", content="能退货吗？"),
            Message(conversation_id=3, role="assistant", content="我先为您查询。"),
            Message(conversation_id=3, role="assistant", content="支持七天无理由退货。"),
        ])
    model = FakeStructuredModel([QASet(items=[])])

    extract_batches(session_factory, model)

    transcript = "\n".join(message.content for message in model.inputs[0])
    assert "能退货吗？" in transcript
    assert "我先为您查询。" not in transcript
    assert "支持七天无理由退货。" in transcript


def test_extraction_rejects_candidate_for_conversation_outside_current_batch(session_factory):
    from app.knowledge.extraction import ExtractedQA, QASet, extract_batches

    model = FakeStructuredModel([QASet(items=[ExtractedQA(
        conversation_id=2, question="运费？", answer="满 99 元包邮。"
    )])])

    with pytest.raises(ValueError, match="outside the current batch"):
        extract_batches(session_factory, model, batch_size=1)

    assert _staging(session_factory) == []


def test_extraction_is_idempotent_for_the_same_source_and_qa(session_factory):
    from app.knowledge.extraction import ExtractedQA, QASet, extract_batches

    answer = QASet(items=[ExtractedQA(
        conversation_id=1, question="退货期限是多久？", answer="支持七天无理由退货。"
    )])
    model = FakeStructuredModel([answer, answer])

    assert extract_batches(session_factory, model) == {"batches": 1, "extracted": 1, "skipped": 0}
    assert extract_batches(session_factory, model) == {"batches": 1, "extracted": 0, "skipped": 1}
    rows = _staging(session_factory)
    assert [(row.source_ref, row.question, row.answer, row.status) for row in rows] == [
        ("conversation/1", "退货期限是多久？", "支持七天无理由退货。", "extracted")
    ]


def test_extraction_rolls_back_a_batch_when_a_later_candidate_is_invalid(session_factory):
    from app.knowledge.extraction import ExtractedQA, QASet, extract_batches

    model = FakeStructuredModel([QASet(items=[
        ExtractedQA(conversation_id=1, question="退货期限是多久？", answer="支持七天无理由退货。"),
        ExtractedQA(conversation_id=99, question="伪造问题", answer="伪造答案"),
    ])])

    with pytest.raises(ValueError, match="outside the current batch"):
        extract_batches(session_factory, model)

    assert _staging(session_factory) == []
