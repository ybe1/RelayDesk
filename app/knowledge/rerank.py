"""Async SiliconFlow reranking with bounded requests and validated responses."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable

import httpx


MAX_DOCUMENTS = 50
MAX_TOP_N = 10
MAX_QUERY_BYTES = 8192
MAX_DOCUMENT_BYTES = 16_384
MAX_TOTAL_DOCUMENT_BYTES = 262_144
_RETRYABLE_STATUS_CODES = {429, 503, 504}


class RerankClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = "BAAI/bge-reranker-v2-m3",
        timeout: float = 30.0,
        max_retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.transport = transport
        self.sleep = sleep

    async def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        if not documents:
            return []
        if not self.api_key:
            raise RuntimeError("RERANK_API_KEY is required before reranking")

        self._validate_input(query, documents, top_n)
        result_count = min(top_n, len(documents))
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": result_count,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
            for attempt in range(self.max_retries + 1):
                response = await client.post(f"{self.base_url}/rerank", headers=headers, json=payload)
                if response.status_code in _RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                    await self.sleep(0.25 * (2**attempt))
                    continue
                response.raise_for_status()
                return self._validate_results(response.json(), len(documents), result_count)

        raise AssertionError("unreachable")

    @staticmethod
    def _validate_input(query: str, documents: list[str], top_n: int) -> None:
        if not isinstance(query, str) or not query.strip() or len(query.encode("utf-8")) > MAX_QUERY_BYTES:
            raise ValueError("rerank input has an invalid query")
        if len(documents) > MAX_DOCUMENTS:
            raise ValueError("rerank input has too many documents")
        if type(top_n) is not int or not 1 <= top_n <= MAX_TOP_N:
            raise ValueError("rerank input has an invalid top_n")
        total_bytes = 0
        for document in documents:
            if not isinstance(document, str) or not document.strip():
                raise ValueError("rerank input has an invalid document")
            size = len(document.encode("utf-8"))
            if size > MAX_DOCUMENT_BYTES:
                raise ValueError("rerank input has an oversized document")
            total_bytes += size
        if total_bytes > MAX_TOTAL_DOCUMENT_BYTES:
            raise ValueError("rerank input has oversized documents")

    @staticmethod
    def _validate_results(payload: object, document_count: int, result_count: int) -> list[tuple[int, float]]:
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ValueError("rerank response has no results list")
        results = payload["results"]
        if len(results) != result_count:
            raise ValueError("rerank response has an invalid result count")

        seen: set[int] = set()
        ranked: list[tuple[int, float]] = []
        for item in results:
            if not isinstance(item, dict):
                raise ValueError("rerank response has a non-object result")
            index = item.get("index")
            score = item.get("relevance_score")
            if type(index) is not int or not 0 <= index < document_count or index in seen:
                raise ValueError("rerank response has an invalid index")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("rerank response has an invalid relevance_score")
            seen.add(index)
            ranked.append((index, float(score)))
        return ranked
