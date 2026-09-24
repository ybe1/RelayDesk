import json

import pytest
from pydantic import ValidationError


async def test_query_transport_sends_current_question_and_disables_thinking(structured_transport, monkeypatch):
    from app.config import settings
    from app.knowledge.query_understanding import understand_query

    monkeypatch.setattr(settings, "rag_category_allowlist", ("配送", "退货"))
    model, requests = structured_transport("QueryUnderstanding", {
        "normalized_query": "邮费多少？", "search_query": "邮费多少 运费 配送费", "category": "配送",
    })
    result = await understand_query(model, "邮费多少？")
    assert result.category == "配送"
    assert result.normalized_query == "邮费多少？"
    body = requests[0]
    assert body["thinking"] == {"type": "disabled"}
    assert "response_format" not in body
    assert body["tools"][0]["function"]["name"] == "QueryUnderstanding"
    assert [message["role"] for message in body["messages"]] == ["system", "user"]
    assert json.loads(body["messages"][1]["content"]) == {"current_question": "邮费多少？"}


@pytest.mark.parametrize("category", ['配送" or id > 0', "未配置分类", ""])
async def test_outside_allowlist_output_is_rejected(category, structured_transport, monkeypatch):
    from app.config import settings
    from app.knowledge.query_understanding import understand_query

    monkeypatch.setattr(settings, "rag_category_allowlist", ("配送",))
    model, _ = structured_transport("QueryUnderstanding", {
        "normalized_query": "邮费多少？", "search_query": "邮费 运费", "category": category,
    })
    with pytest.raises(ValidationError):
        await understand_query(model, "邮费多少？")


async def test_empty_allowlist_accepts_only_none(structured_transport, monkeypatch):
    from app.config import settings
    from app.knowledge.query_understanding import QueryUnderstanding, understand_query

    monkeypatch.setattr(settings, "rag_category_allowlist", ())
    model, _ = structured_transport("QueryUnderstanding", {
        "normalized_query": "有什么政策？", "search_query": "政策", "category": None,
    })
    assert (await understand_query(model, "有什么政策？")).category is None
    with pytest.raises(ValidationError):
        QueryUnderstanding(normalized_query="q", search_query="q", category="配送")


@pytest.mark.parametrize("raw_query", ["", "  ", ["历史消息", "当前问题"], {"history": "秘密"}])
async def test_history_containers_and_empty_question_fail_before_transport(raw_query, structured_transport):
    from app.knowledge.query_understanding import understand_query

    model, requests = structured_transport("QueryUnderstanding", {})
    with pytest.raises(ValueError):
        await understand_query(model, raw_query)
    assert requests == []


@pytest.mark.parametrize("change", [{"normalized_query": " "}, {"search_query": ""}, {"history": "private"}])
def test_query_schema_rejects_blank_fields_and_extra_history(change):
    from app.knowledge.query_understanding import QueryUnderstanding

    with pytest.raises(ValidationError):
        QueryUnderstanding.model_validate({"normalized_query": "q", "search_query": "q", "category": None, **change})


def test_query_evaluator_catches_semantic_loss_expansion_leaks_and_wrong_category():
    from scripts.eval_ch04_prompts import assess_query

    sample = {"category": "配送", "normalized_contains": ["邮费"],
              "search_contains": ["邮费"], "search_any": [["运费", "配送费"]],
              "normalized_forbidden": ["运费"], "forbidden": ["MH-999"]}
    good = {"normalized_query": "邮费？", "search_query": "邮费 运费", "category": "配送"}
    assert assess_query(sample, good) == []
    for field, value in [("normalized_query", "运费？"), ("search_query", "MH-999 邮费"),
                         ("category", None)]:
        assert assess_query(sample, {**good, field: value})


def test_evaluation_gate_requires_all_category_and_safety_cases():
    from scripts.eval_ch04_prompts import summarize

    rows = [{"kind": "query", "passed": True, "category_ok": True, "safety": False}] * 19
    assert summarize(rows + [{"kind": "query", "passed": False, "category_ok": False, "safety": False}])["passed"] is False
    assert summarize(rows + [{"kind": "evidence", "passed": False, "safety": True}])["passed"] is False
    assert summarize(rows + [{"kind": "evidence", "passed": False, "safety": False}])["passed"] is True
