"""Synchronous SiliconFlow embeddings client for the Ch03 dense index."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence

import httpx


VECTOR_DIMENSION = 1024
MAX_BATCH_SIZE = 16
_RETRYABLE_STATUS_CODES = {429, 503, 504}


def validate_vectors(vectors: Sequence[Sequence[float]], expected_count: int) -> list[list[float]]:
    """Return validated vectors, rejecting malformed provider responses."""
    if len(vectors) != expected_count:
        raise ValueError(f"embedding count mismatch: expected {expected_count}, got {len(vectors)}")

    validated: list[list[float]] = []
    for index, vector in enumerate(vectors):
        if len(vector) != VECTOR_DIMENSION:
            raise ValueError(f"embedding {index} has dimension {len(vector)}, expected {VECTOR_DIMENSION}")
        converted = [float(value) for value in vector]
        if not all(math.isfinite(value) for value in converted):
            raise ValueError(f"embedding {index} contains a non-finite value")
        validated.append(converted)
    return validated


class EmbeddingClient:
    """OpenAI-compatible client for SiliconFlow's BAAI/bge-m3 endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = "BAAI/bge-m3",
        timeout: float = 20.0,
        max_retries: int = 2,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.transport = transport
        self.sleep = sleep

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("EMBED_API_KEY is required before embedding text")
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), MAX_BATCH_SIZE):
            vectors.extend(self._embed_batch(texts[start : start + MAX_BATCH_SIZE]))
        return vectors

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts, "encoding_format": "float"}
        headers = {"Authorization": f"Bearer {self.api_key}"}

        with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
            for attempt in range(self.max_retries + 1):
                response = client.post(f"{self.base_url}/embeddings", headers=headers, json=payload)
                if response.status_code in _RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                    self.sleep(0.25 * (2**attempt))
                    continue
                response.raise_for_status()
                return self._ordered_vectors(response.json(), len(texts))

        raise AssertionError("unreachable")

    @staticmethod
    def _ordered_vectors(payload: object, expected_count: int) -> list[list[float]]:
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError("embedding response has no data list")

        response_data = payload["data"]
        ordered: list[list[float] | None] = [None] * expected_count
        for item in response_data:
            if not isinstance(item, dict) or not isinstance(item.get("index"), int):
                raise ValueError("embedding response item has no integer index")
            index = item["index"]
            vector = item.get("embedding")
            if index < 0 or index >= expected_count or ordered[index] is not None or not isinstance(vector, list):
                raise ValueError("embedding response indexes are invalid")
            ordered[index] = vector

        if any(vector is None for vector in ordered):
            raise ValueError("embedding response count does not match request")
        return validate_vectors([vector for vector in ordered if vector is not None], expected_count)
