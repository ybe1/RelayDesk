"""Verify the Ch09 ORM schema against the supplied DDL."""

from sqlalchemy.dialects import mysql

from app.db import models


def _columns(table, expected):
    assert set(table.c.keys()) == set(expected)
    for name, (sql_type, nullable, comment) in expected.items():
        column = table.c[name]
        assert column.type.compile(dialect=mysql.dialect()) == sql_type, name
        assert column.nullable is nullable, name
        assert column.comment == comment, name


def test_ch09_models_exist():
    assert getattr(models, "ReviewQueue", None) is not None
    assert getattr(models, "EvalRun", None) is not None


def test_review_queue_matches_given_ddl():
    table = models.ReviewQueue.__table__
    _columns(table, {
        "id": ("BIGINT UNSIGNED", False, "缺口主键,也是查重命中要返回的 matched_question_id"),
        "normalized_question": ("VARCHAR(512)", False, "标准化后的 FAQ 式问题"),
        "ai_suggested_answer": ("TEXT", True, "模型生成的示例答案,备查"),
        "occurrence_count": ("INTEGER UNSIGNED", False, "出现次数,查重命中累加,越高越该优先补"),
        "review_status": ("ENUM('待审','通过','驳回')", False, "人工审核状态"),
        "approved_answer": ("TEXT", True, "审核通过时补的核准答案,走 ch03 落库流程写回知识库"),
        "created_at": ("DATETIME", False, "首次入队时间"),
        "updated_at": ("DATETIME", False, "更新时间"),
    })
    assert [column.name for column in table.primary_key] == ["id"]
    assert table.c.id.autoincrement is True
    assert table.c.occurrence_count.server_default.arg.text == "1"
    assert table.c.review_status.server_default.arg.text == "'待审'"
    assert table.c.created_at.server_default.arg.text == "CURRENT_TIMESTAMP"
    assert table.c.updated_at.server_default.arg.text == "CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"
    assert {index.name: tuple(column.name for column in index.columns) for index in table.indexes} == {
        "idx_review_status": ("review_status",),
    }
    assert table.dialect_options["mysql"]["engine"] == "InnoDB"
    assert table.dialect_options["mysql"]["charset"] == "utf8mb4"
    assert table.comment == "飞轮待审队列"


def test_eval_runs_matches_given_ddl():
    table = models.EvalRun.__table__
    _columns(table, {
        "id": ("BIGINT UNSIGNED", False, "评估轮次主键"),
        "triggered_by": ("ENUM('定时','手动')", False, "这轮怎么起的:定时任务,或某次改动后手动跑"),
        "dataset_size": ("INTEGER UNSIGNED", False, "这轮跑的评估集条数"),
        "metrics": ("JSON", False, '各指标分数,如 {"recall_at_k":0.82,"mrr":0.71,"faithfulness":0.90}'),
        "created_at": ("DATETIME", False, "跑完落表时间"),
    })
    assert table.c.triggered_by.server_default.arg.text == "'定时'"
    assert {index.name: tuple(column.name for column in index.columns) for index in table.indexes} == {
        "idx_created_at": ("created_at",),
    }
    assert table.comment == "自动化评估流水线轮次结果"


def test_low_confidence_extensions_match_given_ddl():
    table = models.LowConfidenceQuestion.__table__
    assert table.c.retrieved_chunks.type.compile(dialect=mysql.dialect()) == "JSON"
    assert table.c.retrieved_chunks.nullable is True
    assert table.c.retrieved_chunks.comment == "落池时的召回片段快照:Top 几条的原文与分数,审核页展示用;没走检索为 NULL"
    assert table.c.matched_review_id.type.compile(dialect=mysql.dialect()) == "BIGINT UNSIGNED"
    assert table.c.matched_review_id.nullable is True
    assert table.c.matched_review_id.comment == "查重后归并到的缺口,指向 review_queue.id"
    foreign_key, = table.c.matched_review_id.foreign_keys
    assert foreign_key.target_fullname == "review_queue.id"
    assert foreign_key.name == "fk_lcq_review"
    assert foreign_key.ondelete == "SET NULL"
    indexes = {index.name: tuple(column.name for column in index.columns) for index in table.indexes}
    assert indexes["idx_matched_review_id"] == ("matched_review_id",)

