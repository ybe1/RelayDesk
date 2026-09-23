"""Verify the Ch04 ORM schema against Appendix A of the implementation plan."""

from sqlalchemy.dialects import mysql

from app.db.models import FaithCase, LowConfidenceQuestion


def _columns(table, expected):
    assert set(table.c.keys()) == set(expected)
    for name, (sql_type, nullable, comment) in expected.items():
        column = table.c[name]
        assert column.type.compile(dialect=mysql.dialect()) == sql_type, name
        assert column.nullable is nullable, name
        assert column.comment == comment, name


def test_low_confidence_columns_defaults_and_storage():
    table = LowConfidenceQuestion.__table__
    _columns(table, {
        "id": ("BIGINT UNSIGNED", False, "主键"),
        "conversation_id": ("BIGINT UNSIGNED", True, "来源会话"),
        "raw_question": ("TEXT", False, "用户原话,带情绪口语"),
        "source": ("ENUM('retrieval_low_conf','self_check','user_feedback')", False, "入池入口:检索证据低 / 生成自评不足 / 用户反馈未解决"),
        "reason": ("TEXT", True, "判不能的原因,留作复盘"),
        "created_at": ("DATETIME", False, "入池时间"),
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
        "id": ("BIGINT UNSIGNED", False, "主键"),
        "eval_id": ("VARCHAR(16)", False, "评估集题号,如 A43;一题一行"),
        "bucket": ("VARCHAR(24)", False, "题目所属桶:A_policy / B_model / C_colloquial / E_multi"),
        "query": ("VARCHAR(512)", False, "用户问题原文"),
        "strategy": ("VARCHAR(24)", False, "产出这条答案的检索策略"),
        "answer": ("TEXT", False, "被判编造的那版生成答案原文"),
        "reason": ("TEXT", False, "裁判给的理由:编在哪一句"),
        "citations": ("JSON", True, "这一轮喂给模型的 Top-K 证据全集快照:[{n,chunk_id,section_path,question,answer}];答案里的角标 [n] 就是这份列表的序号,答案通常只引用其中两三条;老数据没记为 NULL"),
        "judge_model": ("VARCHAR(64)", True, "判这条的裁判模型"),
        "status": ("ENUM('未解决','已解决','无需解决')", False, "处置状态,人工点按钮改"),
        "seen_count": ("INTEGER UNSIGNED", False, "被判编造的累计次数(跨轮)"),
        "first_seen_at": ("DATETIME", False, "第一次被判编造的时间"),
        "last_seen_at": ("DATETIME", False, "最近一次被判编造的时间"),
        "resolution": ("VARCHAR(300)", True, "处置说明:标已解决要写清怎么解决的,标无需解决要写清为什么不用改;退回未解决时清空。空着的处置在台账上等于没有交代"),
        "resolved_at": ("DATETIME", True, "最近一次被标为已解决/无需解决的时间;复发后仍保留,用来标「复发」"),
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
