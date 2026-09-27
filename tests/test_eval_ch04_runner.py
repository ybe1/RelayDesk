import asyncio
from copy import deepcopy
import json

import pytest

from app.knowledge.query_understanding import QueryUnderstanding
from app.knowledge.retrieval import RetrievalResult


def cases():
    group = {"group_id": "fee", "allowed_section_paths": ["运费"], "expect_points": ["满99元包邮"]}
    return [{"id": "A1", "bucket": "A_policy", "difficulty": "easy", "query": "运费", "expected_groups": [group]},
            {"id": "D1", "bucket": "D_out_of_scope", "difficulty": "hard", "query": "火星", "expected_groups": []}]


async def retrieve(case, strategy):
    return RetrievalResult(QueryUnderstanding(normalized_query=case["query"], search_query=case["query"]),
        [{"id": 8, "rank": 1, "question": "运费", "answer": "满99元包邮", "section_path": "运费", "score": 1}],
        True, "retrieval_ok")


async def test_all_four_retrievals_finish_before_generation_and_failures_keep_metrics(tmp_path):
    from scripts.eval_ch04 import evaluate
    retrieved = []
    async def record_retrieval(case, strategy):
        retrieved.append((case["id"], strategy))
        return await retrieve(case, strategy)
    async def generate(case, result):
        assert len(retrieved) == 8
        if case["bucket"] == "A_policy":
            raise TimeoutError("private key must never enter a report")
        return {"answer": "无法回答", "refused": True}
    path = tmp_path / "rag_eval.json"
    report = await evaluate(cases(), retriever=record_retrieval, generator=generate, report_path=path)
    assert report["meta"]["retrieval_complete"] is True
    assert report["generation"]["status"] == "partial"
    assert set(report["retrieval"]) == {"vector", "bm25", "hybrid", "hybrid_rerank"}
    for strategy in report["retrieval"]:
        assert report["retrieval"][strategy]["overall"] == {"case_count": 1, "completed": 1, "skipped": 0,
                                                            "recall_at_5": 1.0, "mrr": 1.0}
        assert report["evidence_coverage"][strategy]["overall"]["coverage"] == 1
        assert report["generation"]["strategies"][strategy]["buckets"]["D_out_of_scope"]["refusal_rate"] == 1
        assert report["cases"][strategy][0]["generation"]["reason"] == "TimeoutError"
    assert json.loads(path.read_text(encoding="utf-8"))["retrieval"] == report["retrieval"]
    assert "private key" not in path.read_text(encoding="utf-8")
    assert "Recall@5" in path.with_suffix(".txt").read_text(encoding="utf-8")


async def test_retrieval_failure_skips_only_that_strategy_and_never_becomes_zero_recall(tmp_path):
    from scripts.eval_ch04 import evaluate
    async def partial(case, strategy):
        if strategy == "bm25":
            raise ConnectionError("unavailable")
        return await retrieve(case, strategy)
    report = await evaluate(cases(), retriever=partial, report_path=tmp_path / "r.json")
    assert report["retrieval"]["bm25"]["overall"]["mrr"] is None
    assert report["retrieval"]["bm25"]["overall"]["skipped"] == 1
    assert report["retrieval"]["vector"]["overall"]["mrr"] == 1
    assert report["generation"]["status"] == "not_requested"


async def test_hallucination_writes_all_ordered_citations_and_d_is_not_judged(tmp_path):
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import sessionmaker
    from app.db.models import FaithCase
    from scripts.eval_ch04 import evaluate, persist_faith_case
    import tests.ch04_sqlite  # noqa: F401 — MySQL type adapters
    engine = create_engine(f"sqlite:///{tmp_path / 'ledger.db'}")
    FaithCase.__table__.create(engine)
    factory = sessionmaker(engine)
    judged = []
    async def many_hits(case, strategy):
        result = await retrieve(case, strategy)
        result.hits = [{"id": rank, "rank": rank, "question": f"条款{rank}", "answer": "完整证据" * 300,
                        "section_path": "运费"} for rank in [1, 3, 5, 6, 4, 2]]
        return result
    async def generate(case, result):
        return {"answer": "凭空数字", "refused": False}
    async def judge(case, answer, citations):
        judged.append(case["id"])
        return {"faithful": False, "reason": "无依据"}
    def ledger(**payload):
        persist_faith_case(session_factory=factory, **payload)
    report = await evaluate(cases(), retriever=many_hits, generator=generate, judge=judge, ledger=ledger,
                            judge_model="local", report_path=tmp_path / "r.json")
    assert judged == ["A1"]
    with factory() as session:
        row = session.scalar(select(FaithCase))
        assert row.eval_id == "A1"
        assert [item["chunk_id"] for item in row.citations] == [1, 3, 5, 6, 4, 2]
        assert [item["n"] for item in row.citations] == [1, 2, 3, 4, 5, 6]
        assert all(item["answer"] == "完整证据" * 300 for item in row.citations)
        assert row.seen_count == 1
    assert report["generation"]["faithfulness"]["hallucination_rate"] == 1


async def test_ledger_failure_is_report_warning_without_erasing_judgement(tmp_path):
    from scripts.eval_ch04 import evaluate
    async def generate(case, result): return {"answer": "编造", "refused": False}
    async def judge(case, answer, citations): return {"faithful": False, "reason": "无依据"}
    def ledger(**payload): raise RuntimeError("secret database credentials")
    report = await evaluate(cases(), retriever=retrieve, generator=generate, judge=judge, ledger=ledger,
                            report_path=tmp_path / "r.json")
    assert report["cases"]["hybrid_rerank"][0]["faithfulness"]["faithful"] is False
    assert report["warnings"] == [{"stage": "ledger", "id": "A1", "strategy": "hybrid_rerank", "error": "RuntimeError"}]
    assert report["retrieval"]["hybrid_rerank"]["overall"]["mrr"] == 1


async def test_resume_generates_from_saved_retrieval_without_rerunning_or_duplicate_ledger(tmp_path):
    from scripts.eval_ch04 import evaluate
    path = tmp_path / "r.json"
    report = await evaluate(cases(), retriever=retrieve, report_path=path)
    async def forbidden(*args): raise AssertionError("must reuse checkpoint")
    async def generate(case, result): return {"answer": "满99元包邮", "refused": case["id"] == "D1"}
    async def judge(*args): return {"faithful": True, "reason": "有依据"}
    resumed = await evaluate(cases(), retriever=forbidden, generator=generate, judge=judge,
                             resume=report, phase="generation", report_path=path)
    assert resumed["generation"]["status"] == "complete"
    repeated = await evaluate(cases(), retriever=forbidden, generator=forbidden, judge=forbidden,
                              resume=resumed, report_path=path)
    assert repeated["generation"]["status"] == "complete"
    changed = deepcopy(cases())
    changed[0]["query"] = "数据发生变化"
    with pytest.raises(ValueError, match="dataset"):
        await evaluate(changed, retriever=retrieve, resume=report)


async def test_timeouts_are_skipped_low_concurrency_is_bounded_and_retry_is_explicit():
    from scripts.eval_ch04 import evaluate
    active = maximum = 0
    async def slow(case, strategy):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        try:
            await asyncio.sleep(1)
        finally:
            active -= 1
    report = await evaluate(cases(), retriever=slow, concurrency=2, timeout=0.005)
    assert maximum == 2
    assert report["meta"]["retrieval_skipped"] == 8
    async def forbidden(*args): raise AssertionError("no implicit retry")
    unchanged = await evaluate(cases(), retriever=forbidden, resume=report)
    assert unchanged["meta"]["retrieval_skipped"] == 8
    retried = await evaluate(cases(), retriever=retrieve, resume=report, retry_skipped=True)
    assert retried["meta"]["retrieval_skipped"] == 0


async def test_generation_phase_requires_all_retrieval_attempts_first():
    from scripts.eval_ch04 import evaluate
    with pytest.raises(ValueError, match="retrieval"):
        await evaluate(cases(), retriever=retrieve, phase="generation")


def test_generation_sample_balances_bucket_and_difficulty_without_reordering_dataset():
    from scripts.eval_ch04 import select_generation_ids
    rows = [{"id": f"{bucket}-{difficulty}-{i}", "bucket": bucket, "difficulty": difficulty}
            for bucket in ["A_policy", "D_out_of_scope"]
            for difficulty in ["easy", "medium", "hard"] for i in range(2)]
    assert select_generation_ids(rows, 6) == {
        "A_policy-easy-0", "A_policy-medium-0", "A_policy-hard-0",
        "D_out_of_scope-easy-0", "D_out_of_scope-medium-0", "D_out_of_scope-hard-0",
    }


async def test_saved_generation_phase_does_not_require_live_retrieval_services(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from app.knowledge.retrieval import RetrievalDependencies
    from scripts.eval_ch04 import _live, evaluate
    report = await evaluate(cases(), retriever=retrieve)
    def offline():
        raise ConnectionError("Milvus offline; saved evidence must suffice")
    monkeypatch.setattr(RetrievalDependencies, "from_settings", offline)
    args = SimpleNamespace(phase="generation", generation_limit=0, timeout=1, concurrency=1,
                           retry_skipped=False, report=tmp_path / "r.json")
    result = await _live(args, cases(), report)
    assert result["meta"]["retrieval_completed"] == 8
    assert result["generation"]["status"] == "partial"


async def test_checkpoint_survives_a_transient_windows_reader_lock(tmp_path, monkeypatch):
    from pathlib import Path
    from scripts.eval_ch04 import evaluate, save_report
    report = await evaluate(cases(), retriever=retrieve)
    path = tmp_path / "r.json"
    path.write_text('{"previous":true}', encoding="utf-8")
    replace = Path.replace
    locks_remaining = 2
    def briefly_locked(source, target):
        nonlocal locks_remaining
        if target == path and locks_remaining:
            locks_remaining -= 1
            raise PermissionError(13, "Windows reader temporarily prevents replacement")
        return replace(source, target)
    monkeypatch.setattr(Path, "replace", briefly_locked)
    save_report(report, path)
    assert json.loads(path.read_text(encoding="utf-8"))["meta"]["retrieval_completed"] == 8


async def test_permanently_locked_checkpoint_preserves_previous_report(tmp_path, monkeypatch):
    from pathlib import Path
    from scripts.eval_ch04 import evaluate, save_report
    report = await evaluate(cases(), retriever=retrieve)
    path = tmp_path / "r.json"
    path.write_text('{"previous":true}', encoding="utf-8")
    def locked(*args): raise PermissionError(13, "permanent lock")
    monkeypatch.setattr(Path, "replace", locked)
    with pytest.raises(PermissionError):
        save_report(report, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"previous": True}


async def test_http_402_stops_remaining_generation_calls_and_keeps_all_retrieval():
    from scripts.eval_ch04 import evaluate
    class BalanceError(Exception):
        status_code = 402
    calls = 0
    async def unpaid(case, result):
        nonlocal calls
        calls += 1
        raise BalanceError("do not store raw provider message")
    report = await evaluate(cases(), retriever=retrieve, generator=unpaid, concurrency=1)
    assert calls == 1
    assert report["meta"]["retrieval_completed"] == 8
    assert report["meta"]["generation_blocked_reason"] == "upstream_http_402"
    assert all(row["generation"]["status"] == "skipped" for rows in report["cases"].values() for row in rows)
    assert report["cases"]["vector"][0]["generation"]["http_status"] == 402
    assert report["cases"]["bm25"][0]["generation"]["reason"] == "upstream_http_402"


async def test_judge_http_402_preserves_answer_and_blocks_remaining_cloud_calls():
    from scripts.eval_ch04 import evaluate
    class BalanceError(Exception):
        status_code = 402
    async def generate(case, result): return {"answer": "满99元包邮", "refused": False}
    async def judge(*args): raise BalanceError("unpaid")
    report = await evaluate(cases(), retriever=retrieve, generator=generate, judge=judge, concurrency=1)
    online = report["cases"]["hybrid_rerank"]
    assert online[0]["generation"]["status"] == "ok"
    assert online[0]["faithfulness"]["http_status"] == 402
    assert online[1]["generation"]["status"] == "ok"
    assert online[1]["faithfulness"]["reason"] == "not_applicable"


async def test_judging_waits_for_generation_of_all_four_strategies():
    from scripts.eval_ch04 import evaluate
    generated = []
    observed = []
    async def generate(case, result):
        generated.append(case["id"])
        return {"answer": "满99元包邮", "refused": False}
    async def judge(*args):
        observed.append(len(generated))
        return {"faithful": True, "reason": "有依据"}
    report = await evaluate(cases(), retriever=retrieve, generator=generate, judge=judge, concurrency=1)
    assert observed == [8]
    assert report["generation"]["status"] == "complete"


async def test_generation_limit_cannot_start_judging_with_unselected_unfinished_answers():
    from scripts.eval_ch04 import evaluate
    judged = []
    async def generate(case, result): return {"answer": "满99元包邮", "refused": False}
    async def judge(case, answer, citations):
        judged.append(case["id"])
        return {"faithful": True, "reason": "有依据"}
    report = await evaluate(cases(), retriever=retrieve, generator=generate, judge=judge,
                            generation_limit=1, concurrency=1)
    assert sum(row["generation"]["status"] == "ok" for rows in report["cases"].values() for row in rows) == 4
    assert judged == []
    assert report["generation"]["faithfulness"]["completed"] == 0
    assert report["generation"]["status"] == "partial"


@pytest.mark.parametrize("failure_reason", ["TimeoutError", "APIStatusError"])
async def test_generation_limit_cannot_hide_existing_failed_answers_outside_selection(failure_reason):
    from scripts.eval_ch04 import evaluate
    resume = await evaluate(cases(), retriever=retrieve)
    for rows in resume["cases"].values():
        rows[1]["generation"] = {"status": "skipped", "reason": failure_reason}
    judged = []
    async def generate(case, result): return {"answer": "满99元包邮", "refused": False}
    async def judge(case, answer, citations):
        judged.append(case["id"])
        return {"faithful": True, "reason": "有依据"}
    report = await evaluate(cases(), retriever=retrieve, generator=generate, judge=judge, resume=resume,
                            phase="generation", generation_limit=1, concurrency=1)
    assert judged == []
    assert report["generation"]["faithfulness"]["completed"] == 0
    assert report["generation"]["status"] == "partial"


async def test_generation_limit_allows_judging_when_every_answer_is_already_successful():
    from scripts.eval_ch04 import evaluate
    async def generate(case, result): return {"answer": "满99元包邮", "refused": False}
    resume = await evaluate(cases(), retriever=retrieve, generator=generate)
    async def forbidden(*args): raise AssertionError("successful generation must be reused")
    judged = []
    async def judge(case, answer, citations):
        judged.append(case["id"])
        return {"faithful": True, "reason": "有依据"}
    report = await evaluate(cases(), retriever=forbidden, generator=forbidden, judge=judge, resume=resume,
                            phase="generation", generation_limit=1, concurrency=1)
    assert judged == ["A1"]
    assert report["generation"]["status"] == "complete"
