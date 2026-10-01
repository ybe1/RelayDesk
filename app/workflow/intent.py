import json
import re
from typing import Literal
from pydantic import BaseModel, ConfigDict
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.workflow_prompts import INTENT_PROMPT

Intent = Literal["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊"]


class IntentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: Intent


def detect_smalltalk(raw_question: str) -> bool:
    normalized = re.sub(r"[\s！!。．.，,？?～~]+", "", raw_question).lower()
    return normalized in {"你好", "您好", "嗨", "hello", "hi", "谢谢", "谢谢你", "感谢", "多谢", "再见", "拜拜", "晚安"}


def route_intent(intent: Intent) -> str:
    return {"商品咨询": "knowledge", "退款退货": "knowledge",
            "物流": "business", "订单": "business", "售后": "business",
            "投诉": "complaint", "闲聊": "smalltalk"}[intent]


async def classify_intent(model, raw_question: str) -> IntentDecision:
    if detect_smalltalk(raw_question):
        return IntentDecision(intent="闲聊")
    response = await model.ainvoke([SystemMessage(content=INTENT_PROMPT), HumanMessage(content=raw_question)], max_tokens=128)
    try:
        return IntentDecision.model_validate(json.loads(response.content))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid intent JSON") from exc
