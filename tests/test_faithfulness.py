import json

import pytest
from pydantic import ValidationError


def test_answer_prompt_contract_preserves_demo_scope_and_explicit_subquestions():
    """Contract only; model behavior is checked by labeled live remediation."""
    from app.evaluation.faithfulness import ANSWER_PROMPT
    messages = ANSWER_PROMPT.format_messages(payload='{"query":"甲？乙？","citations":[]}')
    system = messages[0].content
    assert "教学演示" in system and "不代表真实平台承诺" in system
    assert "逐项回答" in system and "不能全局拒答" in system
    assert json.loads(messages[-1].content)["query"] == "甲？乙？"


def test_judge_prompt_treats_removed_demo_scope_as_an_unsupported_platform_claim():
    from app.evaluation.faithfulness import FAITHFULNESS_PROMPT
    system = FAITHFULNESS_PROMPT.format_messages(payload="{}")[0].content
    assert "仅用于教学演示" in system
    assert "不代表真实平台承诺" in system
    assert "faithful=false" in system


def test_evidence_prompt_contract_applies_rules_to_each_explicit_subquestion():
    from app.core.rag_prompts import EVIDENCE_CHECK_PROMPT
    messages = EVIDENCE_CHECK_PROMPT.format_messages(payload='{"query":"甲？乙？","hits":[]}')
    system = messages[0].content
    assert "逐项核对" in system
    assert "条件不满足" in system and "不是证据缺失" in system
    assert "演示" in system and "useful=true" in system


async def test_judge_sends_zero_temperature_and_exact_ordered_full_snapshot(structured_transport):
    from app.evaluation.faithfulness import judge_faithfulness
    model, requests = structured_transport("FaithfulnessDecision", {"faithful": False, "reason": "数值无依据"})
    citations = [{"n": 1, "chunk_id": 8, "rank": 1, "section_path": "甲",
                  "question": "运费", "answer_excerpt": "长" * 800},
                 {"n": 2, "chunk_id": 3, "rank": 2, "section_path": "乙",
                  "question": "时效", "answer_excerpt": "三个工作日"}]
    result = await judge_faithfulness(model, "运费多少", "8元", citations)
    assert result.faithful is False
    assert requests[0]["temperature"] == 0
    payload = json.loads(requests[0]["messages"][-1]["content"])
    assert payload == {"query": "运费多少", "answer": "8元", "citations": citations}
    assert requests[0]["thinking"] == {"type": "disabled"}


@pytest.mark.parametrize("output", [{"faithful": "false", "reason": "错"},
                                    {"faithful": True, "reason": " "},
                                    {"faithful": True, "reason": "对", "extra": 1}])
async def test_malformed_judgement_is_an_error_not_an_automatic_pass(structured_transport, output):
    from app.evaluation.faithfulness import judge_faithfulness
    model, _ = structured_transport("FaithfulnessDecision", output)
    with pytest.raises(ValidationError):
        await judge_faithfulness(model, "问", "答", [])


def test_snapshot_keeps_every_hit_in_context_order_and_entire_evidence():
    from app.evaluation.faithfulness import citation_snapshot
    hits = [{"id": i, "rank": rank, "question": f"问题{i}", "answer": "猫" * 900,
             "section_path": f"来源{i}", "score": 0.1}
            for i, rank in [(1, 1), (3, 3), (2, 2)]]
    snapshot = citation_snapshot(hits)
    assert [row["chunk_id"] for row in snapshot] == [1, 3, 2]
    assert [row["rank"] for row in snapshot] == [1, 3, 2]
    assert [row["n"] for row in snapshot] == [1, 2, 3]
    assert all(row["answer"] == "猫" * 900 for row in snapshot)


async def test_mechanical_refusal_does_not_call_remote_generation():
    from app.evaluation.faithfulness import generate_answer
    answer = await generate_answer(None, "未知问题", {"sufficient": False, "hits": [], "reason": "no_hits"})
    assert answer.refused is True
    assert answer.answer


async def test_generation_self_check_handles_irrelevant_but_mechanically_sufficient_retrieval(structured_transport):
    from app.evaluation.faithfulness import generate_answer
    model, requests = structured_transport("GeneratedAnswer", {
        "answer": "当前知识库证据不足，无法可靠回答这个问题。", "refused": True,
    })
    answer = await generate_answer(model, "火星运费", {"sufficient": True, "hits": [
        {"id": 1, "rank": 1, "question": "运费", "answer": "满99元包邮", "section_path": "运费"},
    ]})
    assert answer.refused is True
    assert len(requests) == 1
    assert requests[0]["tools"][0]["function"]["name"] == "GeneratedAnswer"


async def test_generation_uses_only_supplied_evidence():
    import httpx
    from langchain_openai import ChatOpenAI
    from app.evaluation.faithfulness import generate_answer
    requests = []
    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        name = body["tools"][0]["function"]["name"]
        output = {"useful": True, "reason": "有依据"} if name == "EvidenceDecision" else {
            "answer": "满99元包邮[1]", "refused": False,
        }
        return httpx.Response(200, json={"id": "local", "object": "chat.completion", "created": 0,
            "model": "local", "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{"id": "one", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(output)}}]}}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        model = ChatOpenAI(model="local", api_key="local", http_async_client=client, max_retries=0)
        answer = await generate_answer(model, "运费", {"sufficient": True, "hits": [
            {"id": 1, "rank": 1, "question": "运费", "answer": "满99元包邮", "section_path": "运费"},
        ]})
    assert answer.answer == "满99元包邮[1]" and answer.refused is False
    assert len(requests) == 1
    assert json.loads(requests[-1]["messages"][-1]["content"])["citations"][0]["answer"] == "满99元包邮"
