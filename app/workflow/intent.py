import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.workflow_prompts import INTENT_PROMPT

from .routing_contracts import IntentDecision, IntentRun
from .model import BudgetedModel


def detect_smalltalk(raw_question: str) -> bool:
    normalized = re.sub(r"[\s！!。．.，,？?～~]+", "", raw_question).lower()
    return normalized in {"你好", "您好", "嗨", "hello", "hi", "谢谢", "谢谢你", "感谢", "多谢", "再见", "拜拜", "晚安"}


def route_intent(intent: str) -> str:
    return {"商品咨询": "knowledge", "退款退货": "knowledge",
            "物流": "business", "订单": "business", "售后": "business",
            "投诉": "complaint", "闲聊": "smalltalk", "其他": "fallback"}[intent]


async def classify_intent(model, raw_question: str,history=None) -> IntentDecision:
    if detect_smalltalk(raw_question):
        return IntentDecision(intent="闲聊", confidence=1.0)
    payload=raw_question if history is None else json.dumps({'history':[m.model_dump() for m in history],'current_question':raw_question},ensure_ascii=False)
    response = await model.ainvoke([SystemMessage(content=INTENT_PROMPT), HumanMessage(content=payload)],
                                  max_tokens=128,response_format={"type":"json_object"},
                                  extra_body={"thinking":{"type":"disabled"}})
    try:
        return IntentDecision.model_validate(json.loads(response.content))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid intent JSON") from exc


async def classify_with_escalation(large_model, small_model, question, budget, threshold=0.75,history=None):
    if detect_smalltalk(question):
        return IntentRun(IntentDecision(intent='闲聊',confidence=1.0),'rule',False)
    upgraded=False
    if small_model is not None:
        try:
            decision=await classify_intent(BudgetedModel(small_model,budget,128),question,history)
        except ValueError:
            decision=None
        if decision is not None and decision.confidence>=threshold:
            return IntentRun(decision,'small',False)
        upgraded=True
    decision=await classify_intent(BudgetedModel(large_model,budget,128),question,history)
    if decision.confidence<threshold:
        decision=IntentDecision(intent='其他',confidence=decision.confidence)
    return IntentRun(decision,'large',upgraded)
