"""Verify the Ch04 ORM schema against Appendix A of the implementation plan."""

from sqlalchemy.dialects import mysql

from app.db.models import FaithCase, LowConfidenceQuestion


def _columns(table, expected):
    assert set(table.c.keys()) == set(expected)
    for name, (sql_type, nullable) in expected.items():
        column = table.c[name]
        assert column.type.compile(dialect=mysql.dialect()) == sql_type, name
        assert column.nullable is nullable, name
        assert column.comment, name


def test_low_confidence_columns_defaults_and_storage():
    table = LowConfidenceQuestion.__table__
    _columns(table, {
        "id": ("BIGINT UNSIGNED", False),
        "conversation_id": ("BIGINT UNSIGNED", True),
        "raw_question": ("TEXT", False),
        "source": ("ENUM('retrieval_low_conf','self_check','user_feedback')", False),
        "reason": ("TEXT", True),
        "created_at": ("DATETIME", False),
    })
    assert [column.name for column in table.primary_key] == ["id"]
    assert table.c.id.autoincrement is True
    assert table.c.created_at.server_default.arg.text == "CURRENT_TIMESTAMP"
    assert table.dialect_options["mysql"]["engine"] == "InnoDB"
    assert table.dialect_options["mysql"]["charset"] == "utf8mb4"
    assert table.comment == "低置信度问题池"


def test_low_confidence_foreign_key_and_indexes():
    table = LowConfidenceQuestion.__table__
    foreign_key, = table.c.conversation_id.foreign_keys
    assert foreign_key.target_fullname == "conversations.id"
    assert foreign_key.name == "fk_lcq_conversation"
    assert {index.name: tuple(c.name for c in index.columns) for index in table.indexes} == {
        "idx_source": ("source",),
        "idx_created_at": ("created_at",),
    }


def test_faith_case_columns_defaults_and_storage():
    table = FaithCase.__table__
    _columns(table, {
        "id": ("BIGINT UNSIGNED", False),
        "eval_id": ("VARCHAR(16)", False),
        "bucket": ("VARCHAR(24)", False),
        "query": ("VARCHAR(512)", False),
        "strategy": ("VARCHAR(24)", False),
        "answer": ("TEXT", False),
        "reason": ("TEXT", False),
        "citations": ("JSON", True),
        "judge_model": ("VARCHAR(64)", True),
        "status": ("ENUM('未解决','已解决','无需解决')", False),
        "seen_count": ("INTEGER UNSIGNED", False),
        "first_seen_at": ("DATETIME", False),
        "last_seen_at": ("DATETIME", False),
        "resolution": ("VARCHAR(300)", True),
        "resolved_at": ("DATETIME", True),
    })
    assert [column.name for column in table.primary_key] == ["id"]
    assert table.c.id.autoincrement is True
    assert table.c.strategy.server_default.arg.text == "'hybrid_rerank'"
    assert table.c.status.server_default.arg.text == "'未解决'"
    assert table.c.seen_count.server_default.arg.text == "1"
    assert table.c.first_seen_at.server_default.arg.text == "CURRENT_TIMESTAMP"
    assert table.c.last_seen_at.server_default.arg.text == "CURRENT_TIMESTAMP"
    assert table.dialect_options["mysql"]["engine"] == "InnoDB"
    assert table.dialect_options["mysql"]["charset"] == "utf8mb4"
    assert table.comment == "ch04 忠实度编造个案台账"


def test_faith_case_unique_key_and_indexes():
    table = FaithCase.__table__
    assert {index.name: (tuple(c.name for c in index.columns), index.unique) for index in table.indexes} == {
        "uk_eval_id": (("eval_id",), True),
        "idx_status": (("status",), False),
        "idx_last_seen_at": (("last_seen_at",), False),
    }


def test_missing_citation_snapshot_is_sql_null():
    processor = FaithCase.__table__.c.citations.type.bind_processor(mysql.dialect())
    assert processor(None) is None
