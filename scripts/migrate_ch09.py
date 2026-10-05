"""Apply/check the Ch09 review queue, evaluation runs, and pool extensions."""

from __future__ import annotations

import argparse

from sqlalchemy import inspect, text
from sqlalchemy.dialects.mysql import dialect

from app.db.models import EvalRun, LowConfidenceQuestion, ReviewQueue
from app.db.session import engine


def _default(value):
    return str(value).lower().replace("()", "").strip("'") if value is not None else None


def inspect_state(connection) -> dict:
    inspector = inspect(connection)
    tables = set(inspector.get_table_names())
    if "low_confidence_questions" not in tables:
        raise RuntimeError("low_confidence_questions is missing; apply Ch04 first")
    return {
        "tables": tables,
        "low_confidence_columns": {item["name"] for item in inspector.get_columns("low_confidence_questions")},
        "low_confidence_indexes": {item["name"] for item in inspector.get_indexes("low_confidence_questions")},
        "low_confidence_foreign_keys": {item.get("name") for item in inspector.get_foreign_keys("low_confidence_questions")},
    }


def planned_operations(state: dict) -> list[str]:
    operations: list[str] = []
    tables = set(state["tables"])
    columns = set(state["low_confidence_columns"])
    indexes = set(state["low_confidence_indexes"])
    foreign_keys = set(state["low_confidence_foreign_keys"])
    if "review_queue" not in tables:
        operations.append("CREATE TABLE review_queue")
    if "eval_runs" not in tables:
        operations.append("CREATE TABLE eval_runs")
    if "retrieved_chunks" not in columns:
        operations.append("ADD COLUMN retrieved_chunks")
    if "matched_review_id" not in columns:
        operations.append("ADD COLUMN matched_review_id")
    if "idx_matched_review_id" not in indexes:
        operations.append("ADD INDEX idx_matched_review_id")
    if "fk_lcq_review" not in foreign_keys:
        operations.append("ADD CONSTRAINT fk_lcq_review")
    return operations


def _table_shape(connection, name: str) -> dict:
    inspector = inspect(connection)
    columns = inspector.get_columns(name)
    return {
        "columns": {
            column["name"]: (
                str(column["type"].compile(dialect=connection.dialect)).upper(),
                column["nullable"],
                _default(column.get("default")),
                column.get("comment"),
            )
            for column in columns
        },
        "indexes": {
            item["name"]: (item["column_names"], bool(item["unique"]))
            for item in inspector.get_indexes(name)
        },
        "primary_key": inspector.get_pk_constraint(name)["constrained_columns"],
        "foreign_keys": inspector.get_foreign_keys(name),
    }


def _expected_table_shape(table) -> dict:
    return {
        "columns": {
            column.name: (
                str(column.type.compile(dialect=dialect())).upper(),
                column.nullable,
                _default(column.server_default.arg) if column.server_default else None,
                column.comment,
            )
            for column in table.c
        },
        "indexes": {
            index.name: ([column.name for column in index.columns], bool(index.unique))
            for index in table.indexes
        },
        "primary_key": [column.name for column in table.primary_key],
        "foreign_keys": [],
    }


def _validate_table(connection, table) -> None:
    actual = _table_shape(connection, table.name)
    expected = _expected_table_shape(table)
    if actual != expected:
        raise RuntimeError(f"ch09 schema mismatch for {table.name}; existing table not changed")
    row = connection.execute(text(
        "SELECT ENGINE,TABLE_COLLATION,TABLE_COMMENT FROM information_schema.TABLES "
        "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=:name"
    ), {"name": table.name}).one()
    if row[0].upper() != "INNODB" or not row[1].startswith("utf8mb4_") or row[2] != table.comment:
        raise RuntimeError(f"ch09 engine/charset/comment mismatch for {table.name}")


def _validate_pool_extensions(connection) -> None:
    inspector = inspect(connection)
    columns = {item["name"]: item for item in inspector.get_columns("low_confidence_questions")}
    expected = LowConfidenceQuestion.__table__.c
    for name in ("retrieved_chunks", "matched_review_id"):
        actual = columns.get(name)
        column = expected[name]
        wanted = (str(column.type.compile(dialect=dialect())).upper(), column.nullable, column.comment)
        got = None if actual is None else (
            str(actual["type"].compile(dialect=connection.dialect)).upper(),
            actual["nullable"], actual.get("comment"),
        )
        if got != wanted:
            raise RuntimeError(f"ch09 schema mismatch for low_confidence_questions.{name}")
    indexes = {item["name"]: item for item in inspector.get_indexes("low_confidence_questions")}
    if indexes.get("idx_matched_review_id", {}).get("column_names") != ["matched_review_id"]:
        raise RuntimeError("ch09 index mismatch for idx_matched_review_id")
    foreign_keys = {item.get("name"): item for item in inspector.get_foreign_keys("low_confidence_questions")}
    foreign_key = foreign_keys.get("fk_lcq_review")
    if not foreign_key or foreign_key.get("constrained_columns") != ["matched_review_id"] \
            or foreign_key.get("referred_table") != "review_queue" \
            or foreign_key.get("referred_columns") != ["id"] \
            or (foreign_key.get("options") or {}).get("ondelete", "").upper() != "SET NULL":
        raise RuntimeError("ch09 foreign key mismatch for fk_lcq_review")


def validate_schema(connection) -> None:
    _validate_table(connection, ReviewQueue.__table__)
    _validate_table(connection, EvalRun.__table__)
    _validate_pool_extensions(connection)


def apply_operation(connection, operation: str) -> None:
    if operation == "CREATE TABLE review_queue":
        ReviewQueue.__table__.create(connection)
    elif operation == "CREATE TABLE eval_runs":
        EvalRun.__table__.create(connection)
    elif operation == "ADD COLUMN retrieved_chunks":
        connection.execute(text(
            "ALTER TABLE low_confidence_questions ADD COLUMN retrieved_chunks JSON NULL "
            "COMMENT '落池时的召回片段快照:Top 几条的原文与分数,审核页展示用;没走检索为 NULL' AFTER reason"
        ))
    elif operation == "ADD COLUMN matched_review_id":
        connection.execute(text(
            "ALTER TABLE low_confidence_questions ADD COLUMN matched_review_id BIGINT UNSIGNED NULL "
            "COMMENT '查重后归并到的缺口,指向 review_queue.id' AFTER retrieved_chunks"
        ))
    elif operation == "ADD INDEX idx_matched_review_id":
        connection.execute(text(
            "ALTER TABLE low_confidence_questions ADD KEY idx_matched_review_id (matched_review_id)"
        ))
    elif operation == "ADD CONSTRAINT fk_lcq_review":
        connection.execute(text(
            "ALTER TABLE low_confidence_questions ADD CONSTRAINT fk_lcq_review "
            "FOREIGN KEY (matched_review_id) REFERENCES review_queue (id) ON DELETE SET NULL"
        ))
    else:
        raise ValueError(f"unknown migration operation: {operation}")


def migrate(*, apply: bool = False) -> bool:
    with engine.connect() as connection:
        operations = planned_operations(inspect_state(connection))
        if operations and not apply:
            for operation in operations:
                print(f"PENDING: {operation}")
            return False
        if operations:
            connection.execute(text("SET NAMES utf8mb4"))
            for operation in operations:
                apply_operation(connection, operation)
            connection.commit()
        validate_schema(connection)
        print("Ch09 schema is current")
        return True


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    group = result.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--apply", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return 0 if migrate(apply=args.apply) else 1


if __name__ == "__main__":
    raise SystemExit(main())
