"""Single-call customer service tools."""

import json
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


def _production_retriever(query: str) -> list[dict]:
    """Create dense retrieval dependencies only when the FAQ tool is invoked."""
    from app.config import settings
    from app.knowledge.embedding import EmbeddingClient
    from app.knowledge.retrieval import search_knowledge
    from app.knowledge.vector_store import MilvusStore

    embedder = EmbeddingClient(
        base_url=settings.embed_base_url,
        api_key=settings.embed_api_key,
        model=settings.embed_model,
    )
    store = MilvusStore(uri=settings.milvus_uri)
    return search_knowledge(
        query,
        embedder,
        store,
        top_k=settings.rag_top_k,
        min_score=settings.rag_min_score,
    )


def build_tools(
    session_factory,
    conversation_id: int,
    retriever: Callable[[str], list[dict]] | None = None,
):
    """Bind persistence and trusted conversation identity to model-visible tools."""

    @tool(args_schema=OrderArgs)
    def query_order(order_id: str) -> str:
        """查询订单演示信息。订单数据为随机模拟，不能代表真实订单状态。"""
        return json.dumps({"mock": True, "order_id": order_id,
                           "status": random.choice(["待发货", "运输中", "已完成"]),
                           "amount": round(random.uniform(29, 399), 2)}, ensure_ascii=False)

    @tool(args_schema=ProductArgs)
    def query_product(product_name: str) -> str:
        """查询商品演示信息。商品数据为随机模拟。"""
        return json.dumps({"mock": True, "product_name": product_name,
                           "price": round(random.uniform(19, 299), 2),
                           "stock": random.randint(0, 200)}, ensure_ascii=False)

    @tool(args_schema=OrderArgs)
    def query_logistics(order_id: str) -> str:
        """查询指定订单的物流演示信息。物流轨迹为随机模拟。"""
        return json.dumps({"mock": True, "order_id": order_id,
                           "status": random.choice(["已揽收", "运输中", "派送中", "已签收"]),
                           "location": random.choice(["上海分拨中心", "杭州转运中心", "本地配送站"])},
                          ensure_ascii=False)

    @tool(args_schema=FaqArgs)
    def query_faq(keyword: str) -> str:
        """按问题语义检索平台知识库；无命中时返回空结果。"""
        rows = (retriever or _production_retriever)(keyword)
        return json.dumps({"keyword": keyword, "matches": [
            {"question": row["question"], "answer": row["answer"], "category": row["category"]}
            for row in rows]}, ensure_ascii=False)

    @tool(args_schema=TicketArgs)
    def create_ticket(description: str, ticket_type: Literal["售后", "投诉", "咨询"]) -> str:
        """为当前客服会话创建人工工单。会话身份由服务端确定。"""
        from app.repositories.tickets import create_ticket as save_ticket

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
