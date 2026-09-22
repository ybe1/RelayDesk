"""Verify the ORM against the user-supplied Ch03 MySQL contract."""

from sqlalchemy.dialects import mysql
from sqlalchemy.schema import CreateTable

from app.db.models import KnowledgeChunk, QAExtractionStaging


def test_knowledge_columns_match_user_ddl():
    table = KnowledgeChunk.__table__
    expected = {
        "id": ("BIGINT UNSIGNED", False),
        "category": ("VARCHAR(255)", False),
        "questions": ("TEXT", False),
        "answer": ("TEXT", False),
        "section_path": ("VARCHAR(512)", True),
        "content_type": ("VARCHAR(32)", True),
        "is_key_clause": ("TINYINT(1)", False),
        "prev_chunk_id": ("BIGINT UNSIGNED", True),
        "next_chunk_id": ("BIGINT UNSIGNED", True),
        "vector_id": ("VARCHAR(64)", True),
        "vectorize_status": ("ENUM('pending','done')", False),
        "created_at": ("DATETIME", False),
        "updated_at": ("DATETIME", False),
    }
    assert set(table.c.keys()) == set(expected)
    for name, (sql_type, nullable) in expected.items():
        column = table.c[name]
        assert column.type.compile(dialect=mysql.dialect()) == sql_type, name
        assert column.nullable is nullable, name
        assert column.comment, name
    assert [column.name for column in table.primary_key] == ["id"]
    assert table.c.id.autoincrement is True
    assert table.c.vectorize_status.server_default.arg.text == "'pending'"
    assert str(table.c.is_key_clause.server_default.arg) == "0"
    assert table.c.created_at.server_default.arg.text == "CURRENT_TIMESTAMP"
    assert table.c.updated_at.server_default.arg.text == "CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"


def test_knowledge_indexes_foreign_keys_and_storage():
    table = KnowledgeChunk.__table__
    assert {index.name: tuple(c.name for c in index.columns) for index in table.indexes} == {
        "idx_category": ("category",),
        "idx_vectorize_status": ("vectorize_status",),
    }
    for column, constraint in (
        ("prev_chunk_id", "fk_chunks_prev"),
        ("next_chunk_id", "fk_chunks_next"),
    ):
        foreign_key, = table.c[column].foreign_keys
        assert foreign_key.target_fullname == "knowledge_chunks.id"
        assert foreign_key.name == constraint
        assert foreign_key.ondelete == "SET NULL"
    assert table.dialect_options["mysql"]["engine"] == "InnoDB"
    assert table.dialect_options["mysql"]["charset"] == "utf8mb4"
    assert table.comment == "知识库 chunk 原文权威源"
    rendered = str(CreateTable(table).compile(dialect=mysql.dialect()))
    assert "ON UPDATE CURRENT_TIMESTAMP" in rendered
    assert "AUTO_INCREMENT" in rendered


def test_staging_columns_indexes_and_defaults():
    table = QAExtractionStaging.__table__
    expected = {
        "id": ("BIGINT UNSIGNED", False),
        "batch_no": ("VARCHAR(64)", False),
        "source_ref": ("VARCHAR(255)", True),
        "question": ("TEXT", False),
        "answer": ("TEXT", False),
        "status": ("ENUM('extracted','kept','discarded')", False),
        "created_at": ("DATETIME", False),
    }
    assert set(table.c.keys()) == set(expected)
    for name, (sql_type, nullable) in expected.items():
        column = table.c[name]
        assert column.type.compile(dialect=mysql.dialect()) == sql_type, name
        assert column.nullable is nullable, name
        assert column.comment, name
    assert [column.name for column in table.primary_key] == ["id"]
    assert table.c.id.autoincrement is True
    assert table.c.status.server_default.arg.text == "'extracted'"
    assert table.c.created_at.server_default.arg.text == "CURRENT_TIMESTAMP"
    assert {index.name: tuple(c.name for c in index.columns) for index in table.indexes} == {
        "idx_batch_no": ("batch_no",),
        "idx_status": ("status",),
    }
    assert table.dialect_options["mysql"]["engine"] == "InnoDB"
    assert table.dialect_options["mysql"]["charset"] == "utf8mb4"
