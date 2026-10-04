"""Single-call customer service tools."""

import json
import asyncio
import inspect
import random
from collections.abc import Callable
from typing import Literal

from langchain.tools import tool
from pydantic import BaseModel, Field


class OrderArgs(BaseModel):
    order_id: str = Field(min_length=1, description="订单号")


class ProductArgs(BaseModel):
    product_name: str = Field(min_length=1, description="商品名称")


class FaqArgs(BaseModel):
    keyword: str = Field(min_length=1, description="知识库语义检索问题，例如邮费是多少")


class TicketArgs(BaseModel):
    description: str = Field(min_length=1, description="需要人工处理的问题描述")
    ticket_type: Literal["售后", "投诉", "咨询"] = Field(description="工单类型")


def _mock_order_snapshot(order_id: str) -> dict:
    """One reproducible synthetic shipment lifecycle per order; no storage."""
    from app.mock.business import mock_order_snapshot
    return mock_order_snapshot(order_id)


async def _production_retriever(query: str):
    """Create and close hybrid retrieval dependencies for this invocation."""
    from app.knowledge.retrieval import search_knowledge_ch04

    return await search_knowledge_ch04(query)


async def _production_checker(query: str, hits: list[dict]):
    from app.core.llm import get_chat_model
    from app.knowledge.evidence_check import check_evidence

    return await check_evidence(get_chat_model(disable_thinking=True), query, hits)


async def _call(function, *args):
    # Sync legacy retrievers may perform I/O; keep them off the event loop.
    value = await function(*args) if inspect.iscoroutinefunction(function) else await asyncio.to_thread(function, *args)
    return await value if inspect.isawaitable(value) else value


def _bounded_faq_payload(payload: dict, max_bytes: int = 60000) -> str:
    """Bound the actual nested ToolMessage JSON, including escaping overhead."""
    def serialize():
        inner = json.dumps(payload, ensure_ascii=False)
        return inner, len(json.dumps({"result": inner}, ensure_ascii=False).encode("utf-8"))

    def shorten(value, limit):
        if isinstance(value, str):
            return value[:limit]  # Python slices at code point boundaries.
        if isinstance(value, list):
            return [shorten(item, limit) for item in value]
        if isinstance(value, dict):
            return {key: shorten(item, limit) for key, item in value.items()}
        return value

    inner, size = serialize()
    limit = 8192
    while size > max_bytes:
        if limit < 1:
            raise ValueError("FAQ metadata exceeds ToolMessage byte budget")
        shortened = shorten(payload, limit)
        payload.clear()
        payload.update(shortened)
        inner, size = serialize()
        limit //= 2
    return inner


def build_tools(
    session_factory,
    conversation_id: int,
    retriever: Callable | None = None,
    evidence_checker: Callable | None = None,
    *,
    raw_question: str | None = None,
    budget=None,
    model=None,
    ticket_session=None,
):
    """Bind persistence and trusted conversation identity to model-visible tools."""
    if budget is not None:
        from app.workflow.retrieval import retrieve_for_turn, check_for_turn
        if retriever is None:
            async def retriever(query):
                return await retrieve_for_turn(query, budget)
        if evidence_checker is None:
            async def evidence_checker(query, hits):
                if model is None:
                    raise ValueError("budgeted FAQ requires a model")
                return await check_for_turn(query, hits, model, budget)

    @tool(args_schema=OrderArgs)
    def query_order(order_id: str) -> str:
        """查询订单演示信息。订单数据为随机模拟，不能代表真实订单状态。"""
        from app.workflow.order_context import demo_order
        return json.dumps(demo_order(order_id),ensure_ascii=False)

    @tool(args_schema=ProductArgs)
    def query_product(product_name: str) -> str:
        """查询商品演示信息。商品数据为随机模拟。"""
        return json.dumps({"mock": True, "product_name": product_name,
                           "price": round(random.uniform(19, 299), 2),
                           "stock": random.randint(0, 200)}, ensure_ascii=False)

    @tool(args_schema=OrderArgs)
    def query_logistics(order_id: str) -> str:
        """查询指定订单的物流演示信息。物流轨迹为随机模拟。"""
        snapshot = _mock_order_snapshot(order_id)
        return json.dumps({"mock": True, "order_id": order_id,
                           "status": snapshot["logistics_status"], "location": snapshot["location"]},
                          ensure_ascii=False)

    @tool(args_schema=FaqArgs)
    async def query_faq(keyword: str) -> str:
        """检索平台规则并核验证据；仅依证据回答，证据不足时拒答，不承诺到账时间。"""
        from app.knowledge.evidence_check import EvidenceDecision
        from app.knowledge.query_understanding import QueryUnderstanding
        from app.knowledge.retrieval import RetrievalResult
        from app.repositories.low_confidence import insert_low_confidence

        query = raw_question if raw_question is not None else keyword
        result = await _call(retriever or _production_retriever, query)
        legacy = isinstance(result, list)
        if legacy:
            result = RetrievalResult(
                QueryUnderstanding(normalized_query=query, search_query=query),
                result, bool(result), "retrieval_ok" if result else "no_hits",
            )
        if not isinstance(result, RetrievalResult):
            raise TypeError("retriever must return RetrievalResult or list[dict]")

        evidence, citations = [], []
        for row in result.hits:
            chunk_id = row.get("id")
            n = len(citations) + 1 if chunk_id is not None else None
            item = {"n": n, "chunk_id": chunk_id, "section_path": row.get("section_path"),
                    "question": row["question"], "answer": row["answer"], "category": row.get("category", "")}
            evidence.append(item)
            if chunk_id is not None:
                citations.append({key: item[key] for key in ("n", "chunk_id", "section_path", "question")} | {
                    "answer_excerpt": row["answer"][:512],
                })
        payload = {"keyword": keyword, "sufficient": result.sufficient, "reason": result.reason,
                   "evidence": evidence, "citations": citations}
        if legacy:
            payload["matches"] = [{key: row[key] for key in ("question", "answer", "category")} for row in result.hits]
        # Judge the same bounded evidence that the final model will receive.
        # Reserve room for the checker reason (2048 code points, nested JSON escaping).
        _bounded_faq_payload(payload, max_bytes=44000)
        source = "retrieval_low_conf"
        if result.sufficient:
            if evidence_checker is not None or not legacy:
                checked_hits = [{"id": item["chunk_id"], "question": item["question"],
                                 "answer": item["answer"], "category": item["category"]}
                                for item in payload["evidence"]]
                decision = await _call(evidence_checker or _production_checker,
                                       query, checked_hits)
                decision = EvidenceDecision.model_validate(decision)
                payload["sufficient"] = decision.useful
                payload["reason"] = decision.reason[:2048]
                source = "self_check"
        if not payload["sufficient"]:
            payload["evidence"], payload["citations"] = [], []
            if legacy:
                payload["matches"] = []
            # All network/model awaits have completed before opening this transaction.
            try:
                with session_factory() as session:
                    try:
                        insert_low_confidence(session, conversation_id, query,
                                              source, payload["reason"])
                        session.commit()
                    except Exception:
                        session.rollback()
                        raise
            except Exception as exc:
                # A failed/uncertain write must not trigger the read-only retry policy.
                raise RuntimeError("low-confidence persistence failed") from exc
        return _bounded_faq_payload(payload)

    @tool(args_schema=TicketArgs)
    def create_ticket(description: str, ticket_type: Literal["售后", "投诉", "咨询"]) -> str:
        """为当前客服会话创建人工工单。会话身份由服务端确定。"""
        from app.repositories.tickets import create_ticket as save_ticket

        if ticket_session is not None:
            ticket = save_ticket(ticket_session, conversation_id, description, ticket_type)
            return json.dumps({"ticket_no": ticket.ticket_no, "conversation_id": conversation_id,
                               "status": "待处理"}, ensure_ascii=False)
        with session_factory() as session:
            try:
                ticket = save_ticket(session, conversation_id, description, ticket_type)
                ticket_no = ticket.ticket_no
                session.commit()
            except Exception:
                session.rollback()
                raise
        return json.dumps({"ticket_no": ticket_no, "conversation_id": conversation_id,
                           "status": "待处理"}, ensure_ascii=False)

    return [query_order, query_product, query_logistics, query_faq, create_ticket]
