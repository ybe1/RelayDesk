import json

import pytest
from pydantic import ValidationError


async def test_evidence_transport_projects_hits_and_disables_thinking(structured_transport):
    from app.knowledge.evidence_check import check_evidence

    model, requests = structured_transport("EvidenceDecision", {"useful": True, "reason": "明确提供七日时限"})
    result = await check_evidence(model, "退货期限？", [{
        "id": 12, "question": "退货期限", "answer": "签收后七天", "category": "退货",
        "score": 0.9, "private_internal": "do-not-forward",
    }])
    assert result.useful is True
    body = requests[0]
    assert body["thinking"] == {"type": "disabled"}
    assert body["tools"][0]["function"]["name"] == "EvidenceDecision"
    payload = json.loads(body["messages"][1]["content"])
    assert payload == {"query": "退货期限？", "hits": [{
        "id": 12, "question": "退货期限", "answer": "签收后七天", "category": "退货",
    }]}
    assert "do-not-forward" not in json.dumps(body)


async def test_no_hits_is_insufficient_without_model(structured_transport):
    from app.knowledge.evidence_check import check_evidence

    model, requests = structured_transport("EvidenceDecision", {})
    assert (await check_evidence(model, "期限？", [])).useful is False
    assert requests == []


@pytest.mark.parametrize("output", [{"useful": "true", "reason": "wrong boolean"},
                                   {"useful": True, "reason": " "},
                                   {"useful": True, "reason": "ok", "answer": "extra"}])
async def test_invalid_evidence_output_is_rejected(output, structured_transport):
    from app.knowledge.evidence_check import check_evidence

    model, _ = structured_transport("EvidenceDecision", output)
    with pytest.raises(ValidationError):
        await check_evidence(model, "期限？", [{"answer": "七天"}])


@pytest.mark.parametrize("query,hits", [("", []), (["history"], []), ("q", "not-hits"), ("q", ["not-a-hit"])])
async def test_bad_inputs_fail_before_model(query, hits, structured_transport):
    from app.knowledge.evidence_check import check_evidence

    model, requests = structured_transport("EvidenceDecision", {})
    with pytest.raises(ValueError):
        await check_evidence(model, query, hits)
    assert requests == []


def test_evidence_evaluator_requires_boolean_match_and_nonempty_reason():
    from scripts.eval_ch04_prompts import assess_evidence

    assert assess_evidence({"useful": False}, {"useful": False, "reason": "缺少时限"}) == []
    assert assess_evidence({"useful": False}, {"useful": True, "reason": "有相关词"})
    assert assess_evidence({"useful": True}, {"useful": True, "reason": " "})
