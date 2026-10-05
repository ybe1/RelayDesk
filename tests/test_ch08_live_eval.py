from types import SimpleNamespace

from scripts.eval_ch08 import grade_live_ticket, grade_live_tool_call, live_ticket_cases, live_tool_cases


def test_live_dataset_contains_only_synthetic_tool_selection_cases():
    cases = live_tool_cases()
    assert [case["id"] for case in cases] == ["L01", "L02"]
    assert [case["expected_tool"] for case in cases] == ["query_logistics", "query_warranty"]
    assert all("1001" in case["question"] and "session_id" not in case for case in cases)


def test_live_tool_grading_requires_one_correct_call_with_id():
    ok = grade_live_tool_call(SimpleNamespace(tool_calls=[{"name": "query_logistics", "id": "call-1"}]), "query_logistics")
    wrong = grade_live_tool_call(SimpleNamespace(tool_calls=[{"name": "query_order", "id": "call-2"}]), "query_logistics")
    multiple = grade_live_tool_call(SimpleNamespace(tool_calls=[{"name": "query_logistics", "id": "call-1"},
        {"name": "query_order", "id": "call-2"}]), "query_logistics")
    assert ok == {"passed": True, "tool_name": "query_logistics", "tool_call_id": "call-1"}
    assert wrong["passed"] is False
    assert multiple["passed"] is False


def test_live_ticket_grading_distinguishes_clarification_from_preview():
    cases = live_ticket_cases()
    assert [case["expected_action"] for case in cases] == ["clarify", "preview"]
    clarify = grade_live_ticket(SimpleNamespace(tool_calls=[], content="请补充具体问题描述。"), cases[0])
    preview = grade_live_ticket(SimpleNamespace(tool_calls=[{"name":"create_ticket","id":"ticket-1",
        "args":{"description":"订单1001包裹破损","ticket_type":"售后"}}],content=""), cases[1])
    bad_preview = grade_live_ticket(SimpleNamespace(tool_calls=[{"name":"create_ticket","id":"ticket-2",
        "args":{"description":"订单已丢失","ticket_type":"售后"}}],content=""), cases[1])
    assert clarify["passed"] is True
    assert preview["passed"] is True
    assert bad_preview["passed"] is False
