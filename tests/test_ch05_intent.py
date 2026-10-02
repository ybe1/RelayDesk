import pytest
from pydantic import ValidationError


def test_smalltalk_matches_entire_short_phrase_only():
    from app.workflow.intent import detect_smalltalk
    for text in ["你好", "谢谢！", "再见", "您好。"]:
        assert detect_smalltalk(text)
    for text in ["你好，订单1001到哪了", "谢谢但是我要投诉", "退款", ""]:
        assert not detect_smalltalk(text)


def test_seven_routes_are_fixed_and_schema_rejects_injected_routes():
    from app.workflow.intent import IntentDecision, route_intent
    expected = {"商品咨询": "knowledge", "退款退货": "knowledge", "物流": "business",
                "订单": "business", "售后": "business", "投诉": "complaint", "闲聊": "smalltalk"}
    for intent, route in expected.items():
        assert route_intent(IntentDecision(intent=intent,confidence=0.95).intent) == route
    with pytest.raises(ValidationError):
        IntentDecision(intent="知识")
    with pytest.raises(ValidationError):
        IntentDecision(intent="闲聊", route="business")


async def test_classifier_reads_valid_json_and_rejects_invalid_output():
    from langchain_core.messages import AIMessage
    from tests.ch05_fakes import SequenceModel
    from app.workflow.intent import classify_intent
    result = await classify_intent(SequenceModel([AIMessage(content='{"intent":"物流","confidence":0.95}')]), "物流呢")
    assert result.intent == "物流"
    with pytest.raises(ValueError):
        await classify_intent(SequenceModel([AIMessage(content='{"intent":"删库"}')]), "问题")
