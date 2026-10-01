"""Reuse chapter 4 retrieval with a shared turn budget and a pre-Agent gate."""
import asyncio
import math
from time import monotonic
from dataclasses import dataclass
from .model import BudgetedModel


@dataclass
class GateDecision:
    allowed: bool
    reason: str


def gate_result(result, threshold):
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("gate threshold must be finite and positive")
    if not result.sufficient or not result.hits:
        return GateDecision(False, result.reason or "no_hits")
    scores = [hit.get("rerank_score") for hit in result.hits]
    if any(not isinstance(s, (int, float)) or isinstance(s, bool) or not math.isfinite(s) for s in scores):
        return GateDecision(False, "invalid_rerank_score")
    return GateDecision(max(scores) >= threshold, "evidence_ok" if max(scores) >= threshold else "evidence_below_threshold")


class CitationRegistry:
    def __init__(self, citations=None):
        self.citations = list(citations or [])

    def merge(self, hits):
        index = {item["chunk_id"]: item["n"] for item in self.citations}
        evidence = []
        for hit in hits:
            cid = hit.get("id", hit.get("chunk_id"))
            if cid is None:
                continue
            if cid not in index:
                n = len(self.citations) + 1
                index[cid] = n
                self.citations.append({"n": n, "chunk_id": cid, "section_path": hit.get("section_path"),
                    "question": hit.get("question", ""), "answer_excerpt": hit.get("answer", "")[:512]})
            evidence.append({**hit, "n": index[cid], "chunk_id": cid})
        return evidence


async def retrieve_for_turn(query, budget):
    from app.knowledge.retrieval import RetrievalDependencies, search_knowledge_ch04
    owned = None
    try:
        budget.check_deadline()
        async with asyncio.timeout(budget.remaining_seconds()):
            owned = await RetrievalDependencies.afrom_settings(max_retries=0)
        owned.model = BudgetedModel(owned.model, budget, output_cap=256)
        if hasattr(owned, "embedder"):
            owned.embedder = MeasuredEmbedder(owned.embedder, budget)
        if hasattr(owned, "reranker"):
            owned.reranker = MeasuredReranker(owned.reranker, budget)
        async with asyncio.timeout(budget.remaining_seconds()):
            return await search_knowledge_ch04(query, dependencies=owned)
    finally:
        if owned is not None:
            await owned.aclose()


class MeasuredEmbedder:
    def __init__(self, inner, budget): self.inner, self.budget = inner, budget
    def embed(self, texts):
        started = monotonic()
        try: return self.inner.embed(texts)
        finally: self.budget.record_external("embedding", monotonic()-started)


class MeasuredReranker:
    def __init__(self, inner, budget): self.inner, self.budget = inner, budget
    async def rerank(self, *args, **kwargs):
        started = monotonic()
        try: return await self.inner.rerank(*args, **kwargs)
        finally: self.budget.record_external("rerank", monotonic()-started)
    async def aclose(self): await self.inner.aclose()


async def check_for_turn(query, hits, model, budget):
    from app.knowledge.evidence_check import check_evidence
    return await check_evidence(BudgetedModel(model, budget, output_cap=256), query, hits)
