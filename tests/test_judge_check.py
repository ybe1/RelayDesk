import json

import httpx
import pytest
from langchain_openai import ChatOpenAI


def case(status="已解决", **changes):
    return {"id": 1, "eval_id": "A1", "query": "运费", "answer": "明天必到",
            "citations": [{"n": 1, "chunk_id": 3, "answer": "证据" * 900}],
            "status": status, "resolution": "人工处置", **changes}


async def run_with_responses(cases, outputs):
    from scripts.judge_check import replay_cases
    requests = []
    iterator = iter(outputs)
    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        value = next(iterator)
        if isinstance(value, Exception):
            raise value
        message = {"role": "assistant", "content": ""}
        if value is not None:
            message["tool_calls"] = [{"id": "local", "type": "function", "function": {
                "name": "FaithfulnessDecision", "arguments": json.dumps(value)}}]
        return httpx.Response(200, json={"id": "local", "object": "chat.completion", "created": 0,
            "model": "local", "choices": [{"index": 0, "finish_reason": "stop", "message": message}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        model = ChatOpenAI(model="local", api_key="local", http_async_client=client, max_retries=0)
        result = await replay_cases(cases, model)
    return result, requests


async def test_status_expectations_stored_snapshot_temperature_and_mismatch():
    rows = [case("未解决"), case("已解决", id=2), case("无需解决", id=3)]
    result, requests = await run_with_responses(rows, [{"faithful": False, "reason": "无依据"},
        {"faithful": False, "reason": "修复不改变旧快照"}, {"faithful": False, "reason": "误判"}])
    assert result["checked"] == 3 and result["mismatches"] == 1 and result["exit_code"] == 1
    assert [row["expected"] for row in result["cases"]] == [False, False, True]
    assert json.loads(requests[0]["messages"][-1]["content"]) == {
        "query": rows[0]["query"], "answer": rows[0]["answer"], "citations": rows[0]["citations"]}
    assert all(request["temperature"] == 0 for request in requests)
    prompt = requests[0]["messages"][0]["content"]
    assert "边界例6" in prompt and "faithful=true" in prompt and "faithful=false" in prompt


async def test_one_empty_completion_retries_then_checks():
    result, requests = await run_with_responses([case()], [None, {"faithful": False, "reason": "无证据"}])
    assert result["checked"] == 1 and result["excluded"] == 0 and result["exit_code"] == 0
    assert len(requests) == 2


async def test_persistent_empty_and_network_failure_excluded_from_denominator():
    result, requests = await run_with_responses([case(), case(id=2), case(id=3)],
        [None, None, httpx.ReadTimeout("secret upstream details"), {"faithful": False, "reason": "无证据"}])
    assert result["checked"] == 1 and result["excluded"] == 2
    assert result["agreement_rate"] == 1.0 and result["exit_code"] == 2
    assert "secret" not in json.dumps(result)
    assert len(requests) == 4


async def test_absent_snapshot_is_excluded_empty_evidence_is_valid():
    result, requests = await run_with_responses([case(citations=None), case(citations=[])],
        [{"faithful": False, "reason": "无证据"}])
    assert result["checked"] == 1 and result["excluded"] == 1 and len(requests) == 1


async def test_no_cases_is_blocked_not_passed():
    result, requests = await run_with_responses([], [])
    assert result["exit_code"] == 2 and result["agreement_rate"] is None and requests == []


def test_cli_inspection_does_not_construct_model(monkeypatch, capsys):
    from scripts import judge_check
    monkeypatch.setattr(judge_check, "load_cases", lambda: [case()])
    monkeypatch.setattr(judge_check, "get_model", lambda: pytest.fail("read-only inspection cannot call model"))
    assert judge_check.main(["--inspect-only"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "blocked" and result["reviewed"] == 1


def test_cli_returns_mismatch_exit_code(monkeypatch, capsys):
    from scripts import judge_check
    monkeypatch.setattr(judge_check, "load_cases", lambda: [case()])
    monkeypatch.setattr(judge_check, "get_model", lambda: object())
    async def replay(rows, model):
        return {"exit_code": 1, "mismatches": 1}
    monkeypatch.setattr(judge_check, "replay_cases", replay)
    assert judge_check.main([]) == 1
    assert json.loads(capsys.readouterr().out)["mismatches"] == 1
