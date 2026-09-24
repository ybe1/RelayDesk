from langchain_core.messages import HumanMessage

from app.core.prompts import CUSTOMER_SERVICE_PROMPT, EXTRACT_PROMPT

def test_customer_service_prompt_renders_with_history():
    msgs = CUSTOMER_SERVICE_PROMPT.format_messages(
        history=[HumanMessage("你们卖猫粮吗?")]
    )
    assert msgs[0].type == "system"
    assert msgs[-1].content == "你们卖猫粮吗?"
    # 行为约束必须写进 system prompt
    for keyword in ("客服", "不", "订单"):
        assert keyword in msgs[0].content


def test_customer_service_prompt_requires_faq_lookup_for_policy_questions():
    system = CUSTOMER_SERVICE_PROMPT.format_messages(history=[])[0].content
    assert "query_faq" in system
    assert "邮费" in system
    assert "知识库语义检索" in system
    assert "演示数据" in system

def test_extract_prompt_renders_text():
    msgs = EXTRACT_PROMPT.format_messages(text="订单 MH1 坏了要退款")
    assert msgs[0].type == "system"
    assert "订单 MH1 坏了要退款" in msgs[-1].content


def test_faq_prompt_requires_evidence_citations_refusal_and_no_arrival_promises():
    system = CUSTOMER_SERVICE_PROMPT.format_messages(history=[])[0].content
    for rule in ("sufficient", "evidence", "[n]", "reason", "到账时间", "不得编造引用"):
        assert rule in system
