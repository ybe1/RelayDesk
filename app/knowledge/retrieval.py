"""Dense online retrieval over the Milvus knowledge read model."""

from __future__ import annotations

from typing import Protocol


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
