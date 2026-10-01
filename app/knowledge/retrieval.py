"""Dense online retrieval over the Milvus knowledge read model."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.knowledge.hybrid_store import HybridMilvusStore, build_filter
from app.knowledge.query_understanding import QueryUnderstanding, understand_query
from app.knowledge.rerank import MAX_DOCUMENT_BYTES, MAX_TOTAL_DOCUMENT_BYTES, RerankClient

_CLEANUP_TASKS: set[asyncio.Task] = set()


def _cleanup_finished(task):
    _CLEANUP_TASKS.discard(task)
    if not task.cancelled() and (error := task.exception()) is not None:
        logging.getLogger(__name__).warning("Retrieval cleanup failed: %s", type(error).__name__)


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class KnowledgeStore(Protocol):
    def search(self, vector: list[float], top_k: int) -> list[dict]: ...


def search_knowledge(
    query: str,
    embedder: Embedder,
    store: KnowledgeStore,
    top_k: int = 5,
    min_score: float = 0.55,
) -> list[dict]:
    """Embed one query, search Milvus, then retain qualified ranked hits."""
    vector = embedder.embed([query])[0]
    hits = store.search(vector, top_k=top_k)
    return sorted(
        (hit for hit in hits if hit["score"] >= min_score),
        key=lambda hit: hit["score"],
        reverse=True,
    )


@dataclass
class RetrievalResult:
    """Mechanical retrieval decision; semantic evidence judgment is a later step."""

    understood_query: QueryUnderstanding
    hits: list[dict]
    sufficient: bool
    reason: str


@dataclass
class RetrievalDependencies:
    """Caller owns injected dependencies; use async with or finally await aclose()."""

    model: Any
    embedder: Embedder
    store: HybridMilvusStore
    reranker: RerankClient
    rerank_min_score: float = 0.0
    _owns_store: bool = field(default=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @classmethod
    def from_settings(cls, *, max_retries=None) -> RetrievalDependencies:
        from app.config import settings
        from app.core.llm import get_chat_model
        from app.knowledge.embedding import EmbeddingClient

        return cls(
            model=get_chat_model(disable_thinking=True, max_retries=max_retries),
            embedder=EmbeddingClient(base_url=settings.embed_base_url, api_key=settings.embed_api_key,
                                     model=settings.embed_model),
            reranker=RerankClient(base_url=settings.rerank_base_url, api_key=settings.rerank_api_key,
                                  model=settings.rerank_model),
            rerank_min_score=settings.rerank_min_score,
            _owns_store=True,
            # Open the connection last; earlier constructor errors cannot leak it.
            store=HybridMilvusStore(uri=settings.milvus_uri, collection_name=settings.milvus_collection),
        )

    @classmethod
    async def afrom_settings(cls, *, max_retries=None) -> RetrievalDependencies:
        """Construct off-loop; a timed-out caller still owns late-result cleanup."""
        initialization = asyncio.create_task(asyncio.to_thread(cls.from_settings, **({"max_retries": max_retries} if max_retries is not None else {})))
        try:
            return await asyncio.shield(initialization)
        except asyncio.CancelledError:
            async def close_late_result():
                try:
                    owned = await initialization
                except Exception:
                    return  # No dependency was returned; consume the late failure.
                await owned.aclose()
            cleanup = asyncio.create_task(close_late_result())
            _CLEANUP_TASKS.add(cleanup)
            cleanup.add_done_callback(_cleanup_finished)
            raise

    async def __aenter__(self) -> RetrievalDependencies:
        if self._closed:
            raise RuntimeError("retrieval dependencies are closed")
        return self

    async def __aexit__(self, _exc_type, _exc, _tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self.reranker.aclose()
        finally:
            if self._owns_store:
                await asyncio.to_thread(self.store.client.close)


async def search_knowledge_ch04(
    query: str,
    *,
    strategy: str = "hybrid_rerank",
    top_k: int = 10,
    category: str | None = None,
    dependencies: RetrievalDependencies | None = None,
    understood_query: QueryUnderstanding | None = None,
) -> RetrievalResult:
    """Four strategies share query understanding, typed filters and context ordering.

    Each recall leg requests 50. Hybrid candidates are Milvus RRF output, deduplicated
    before the optional Top-10 rerank. Operational exceptions always propagate.
    Injected dependencies remain open until their owner closes them.
    """
    from app.config import settings

    if strategy not in {"dense", "vector", "bm25", "hybrid", "hybrid_rerank"}:
        raise ValueError("unsupported retrieval strategy")
    if type(top_k) is not int or not 1 <= top_k <= 10:
        raise ValueError("top_k must be an integer from 1..10")
    if category is not None and (not isinstance(category, str) or category not in settings.rag_category_allowlist):
        raise ValueError("category is outside the configured allowlist")
    if dependencies is None:
        async with await RetrievalDependencies.afrom_settings() as owned:
            return await search_knowledge_ch04(query, strategy=strategy, top_k=top_k,
                                               category=category, dependencies=owned,
                                               understood_query=understood_query)
    if dependencies._closed:
        raise RuntimeError("retrieval dependencies are closed")

    # Offline comparisons can reuse the same validated interpretation for all legs.
    # Normal online calls still perform understanding here exactly once.
    understood = (QueryUnderstanding.model_validate(understood_query.model_dump())
                  if understood_query is not None else await understand_query(dependencies.model, query))
    filter_expr = build_filter(category=category if category is not None else understood.category)
    store = dependencies.store
    if strategy == "bm25":
        candidates = await asyncio.to_thread(store.bm25_search, understood.search_query,
                                             limit=50, filter_expr=filter_expr)
    else:
        vectors = await asyncio.to_thread(dependencies.embedder.embed, [understood.normalized_query])
        if strategy in {"dense", "vector"}:
            candidates = await asyncio.to_thread(store.dense_search, vectors[0], limit=50,
                                                 filter_expr=filter_expr)
        else:
            candidates = await asyncio.to_thread(store.hybrid_search, vectors[0], understood.search_query,
                                                 recall_limit=50, output_limit=50, filter_expr=filter_expr)

    unique: dict[int, dict] = {}
    for candidate in candidates:
        unique.setdefault(candidate["id"], candidate)
    candidates = list(unique.values())[:50]
    if not candidates:
        return RetrievalResult(understood, [], False, "no_hits")

    sufficient, reason = True, "retrieval_ok"
    if strategy == "hybrid_rerank":
        # Share the aggregate budget so every candidate retains a document.
        budget = min(MAX_DOCUMENT_BYTES, MAX_TOTAL_DOCUMENT_BYTES // len(candidates))
        documents = [f"问题：{hit['question']}\n答案：{hit['answer']}".encode("utf-8")[:budget].decode("utf-8", errors="ignore")
                     for hit in candidates]
        scores = await dependencies.reranker.rerank(understood.normalized_query, documents, top_n=top_k)
        ranked = sorted(scores, key=lambda item: item[1], reverse=True)
        hits = [{**candidates[index], "rerank_score": score} for index, score in ranked]
        if hits[0]["rerank_score"] < dependencies.rerank_min_score:
            sufficient, reason = False, "rerank_below_threshold"
    else:
        hits = candidates[:top_k]

    # Persist relevance rank before reordering context; tied scores cannot recover it.
    hits = [{**hit, "rank": rank} for rank, hit in enumerate(hits, 1)]
    ordered = hits[::2] + hits[1::2][::-1]
    return RetrievalResult(understood, ordered, sufficient, reason)
