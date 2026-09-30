import json

import pytest


@pytest.fixture
def stored_report(tmp_path):
    row = {"id": "ch04-E-041", "query": "两件事？", "bucket": "E_multi",
           "expected_groups": [{"expect_points": ["规则甲", "规则乙"]}],
           "retrieval": {"status": "ok", "sufficient": True, "hits": [
               {"id": 8, "rank": 2, "question": "乙", "answer": "规则乙", "section_path": "乙"},
               {"id": 9, "rank": 1, "question": "甲", "answer": "规则甲", "section_path": "甲"}]},
           "generation": {"answer": "旧回答", "refused": True},
           "faithfulness": {"faithful": False, "reason": "遗漏"}}
    path = tmp_path / "rag_eval.json"
    path.write_text(json.dumps({"cases": {"hybrid_rerank": [row, {**row, "id": "ch04-A-001"}]}}), encoding="utf-8")
    path.with_suffix(".txt").write_text("original full report", encoding="utf-8")
    return path


async def test_replays_only_selected_ordered_snapshot_and_keeps_original_reports(stored_report):
    from scripts.remediate_ch04 import remediate
    before = stored_report.read_bytes(), stored_report.with_suffix(".txt").read_bytes()
    calls = []

    async def generator(query, retrieval):
        calls.append(query)
        assert [hit["id"] for hit in retrieval["hits"]] == [8, 9]
        return {"answer": "规则甲；规则乙", "refused": False}

    async def judge(query, answer, citations):
        assert [hit["chunk_id"] for hit in citations] == [8, 9]
        assert [hit["rank"] for hit in citations] == [2, 1]
        return {"faithful": True, "reason": "依据齐全"}

    report, output = await remediate(stored_report, ["ch04-E-041"], generator, judge, model_name="local")
    assert calls == ["两件事？"]
    assert report["summary"] == {"total": 1, "passed": 1, "all_passed": True}
    assert report["rows"][0]["generation"]["answer_point_coverage"] == 1
    assert report["rows"][0]["original_generation"]["answer"] == "旧回答"
    assert output.parent == stored_report.parent / "remediation"
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert before == (stored_report.read_bytes(), stored_report.with_suffix(".txt").read_bytes())


@pytest.mark.parametrize("selection", [[], ["missing"], ["ch04-E-041", "ch04-E-041"], ["ch04-A-001"]])
async def test_invalid_selection_stops_before_remote_work(stored_report, selection):
    from scripts.remediate_ch04 import remediate
    with pytest.raises(ValueError):
        await remediate(stored_report, selection, None, None, model_name="local")
    assert not (stored_report.parent / "remediation").exists()


@pytest.mark.parametrize("answer,refused,faithful", [("规则甲", False, True),
                                                   ("规则甲；规则乙", True, True),
                                                   ("规则甲；规则乙", False, False)])
async def test_success_requires_all_points_no_refusal_and_existing_judge(stored_report, answer, refused, faithful):
    from scripts.remediate_ch04 import remediate
    async def generator(query, retrieval):
        return {"answer": answer, "refused": refused}
    async def judge(query, answer, citations):
        return {"faithful": faithful, "reason": "评估"}
    report, _ = await remediate(stored_report, ["ch04-E-041"], generator, judge, model_name="local")
    assert report["summary"]["all_passed"] is False
    assert report["summary"]["passed"] == 0


async def test_remote_failure_is_saved_without_sensitive_error_body(stored_report):
    from scripts.remediate_ch04 import remediate
    async def generator(query, retrieval):
        raise RuntimeError("secret credential must not leak")
    report, output = await remediate(stored_report, ["ch04-E-041"], generator, None, model_name="local")
    assert report["rows"][0]["error"] == "RuntimeError"
    assert report["summary"]["all_passed"] is False
    assert "secret credential" not in output.read_text(encoding="utf-8")
