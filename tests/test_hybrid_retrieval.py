"""Ch04 orchestration contracts; remote calls replaced at their boundaries."""

from types import SimpleNamespace

import httpx
import pytest

from app.knowledge import retrieval
from app.knowledge.hybrid_store import SearchFilter
from app.knowledge.rerank import RerankClient


def hit(index, score=0.01):
    return {"id": index, "score": score, "question": f"型号{index}", "answer": "说明",
            "category": "商品", "section_path": "目录", "content_type": "faq", "is_key_clause": False}


class Store:
    def __init__(self, hits):
        self.hits, self.calls = hits, []

    def dense_search(self, vector, **kwargs):
        self.calls.append(("dense", vector, kwargs))
        return self.hits

    def bm25_search(self, text, **kwargs):
        self.calls.append(("bm25", text, kwargs))
        return self.hits

    def hybrid_search(self, vector, text, **kwargs):
        self.calls.append(("hybrid", (vector, text), kwargs))
        return self.hits


class Embedder:
    def __init__(self):
        self.inputs = []

    def embed(self, texts):
        self.inputs.append(texts)
        return [[0.25, 0.75]]


@pytest.fixture
def setup(structured_transport, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "rag_category_allowlist", ("商品", "售后"))

    def make(hits=None, response=None, category="商品", threshold=0.5):
        model, model_calls = structured_transport("QueryUnderstanding", {
            "normalized_query": "AX-100 使用", "search_query": "AX-100 使用 操作", "category": category,
        })
        requests = []

        def respond(request):
            import json
            payload = json.loads(request.content)
            requests.append(payload)
            if response is not None:
                return response
            return httpx.Response(200, json={"results": [
                {"index": i, "relevance_score": 0.95 - i * 0.01} for i in range(payload["top_n"])
            ]})

        reranker = RerankClient(base_url="https://local.invalid/v1", api_key="test",
                                max_retries=0, transport=httpx.MockTransport(respond))
        deps = retrieval.RetrievalDependencies(model=model, embedder=Embedder(),
                                               store=Store(hits if hits is not None else [hit(i) for i in range(12)]),
                                               reranker=reranker, rerank_min_score=threshold)
        return deps, requests, model_calls
    return make


@pytest.mark.parametrize("strategy,branch", [("dense", "dense"), ("vector", "dense"),
                                             ("bm25", "bm25"), ("hybrid", "hybrid"),
                                             ("hybrid_rerank", "hybrid")])
async def test_strategy_uses_correct_top50_leg_and_only_reranks_when_requested(setup, strategy, branch):
    deps, requests, _ = setup()
    async with deps:
        result = await retrieval.search_knowledge_ch04("AX-100咋用", strategy=strategy, dependencies=deps)
        assert result.sufficient
        assert result.understood_query.normalized_query == "AX-100 使用"
        assert [x["id"] for x in result.hits] == [0, 2, 4, 6, 8, 9, 7, 5, 3, 1]
        name, query, params = deps.store.calls[0]
        assert name == branch
        assert params["filter_expr"] == SearchFilter(category="商品")
        if branch == "hybrid":
            assert query == ([0.25, 0.75], "AX-100 使用 操作")
            assert params["recall_limit"] == 50
            assert params["output_limit"] == 50
        else:
            assert params["limit"] == 50
        assert deps.embedder.inputs == ([] if branch == "bm25" else [["AX-100 使用"]])
        if branch == "bm25":
            assert query == "AX-100 使用 操作"
        assert len(requests) == (1 if strategy == "hybrid_rerank" else 0)
        if requests:
            assert requests[0]["query"] == "AX-100 使用"
            assert requests[0]["top_n"] == 10
            assert requests[0]["documents"][0] == "问题：型号0\n答案：说明"


async def test_explicit_category_is_validated_and_overrides_understanding(setup):
    deps, _, _ = setup()
    async with deps:
        await retrieval.search_knowledge_ch04("型号", strategy="bm25", category="售后", dependencies=deps)
        assert deps.store.calls[0][2]["filter_expr"] == SearchFilter(category="售后")
        with pytest.raises(ValueError):
            await retrieval.search_knowledge_ch04("型号", category='商品" or id > 0', dependencies=deps)


async def test_null_category_does_not_add_filter(setup):
    deps, _, _ = setup(category=None)
    async with deps:
        await retrieval.search_knowledge_ch04("型号", strategy="bm25", dependencies=deps)
        assert deps.store.calls[0][2]["filter_expr"] == SearchFilter()


async def test_rrf_deduplicates_chunk_ids_before_rerank_and_preserves_metadata(setup):
    deps, requests, _ = setup([hit(7), hit(7), hit(9)])
    async with deps:
        result = await retrieval.search_knowledge_ch04("型号", dependencies=deps)
        assert len(requests[0]["documents"]) == 2
        assert [x["id"] for x in result.hits] == [7, 9]
        assert result.hits[0]["section_path"] == "目录"
        assert result.hits[0]["score"] == 0.01
        assert result.hits[0]["rerank_score"] == 0.95


@pytest.mark.parametrize("scores,sufficient", [([0.5, 0.1], True), ([0.49, 0.1], False)])
async def test_mechanical_threshold_uses_highest_rerank_score_and_keeps_full_evidence(setup, scores, sufficient):
    response = httpx.Response(200, json={"results": [
        {"index": 1, "relevance_score": scores[1]}, {"index": 0, "relevance_score": scores[0]},
    ]})
    deps, _, _ = setup([hit(0), hit(1)], response)
    async with deps:
        result = await retrieval.search_knowledge_ch04("型号", dependencies=deps)
        assert result.sufficient is sufficient
        assert [x["id"] for x in result.hits] == [0, 1]
        assert result.reason == ("retrieval_ok" if sufficient else "rerank_below_threshold")


@pytest.mark.parametrize("strategy", ["dense", "bm25", "hybrid", "hybrid_rerank"])
async def test_empty_recall_is_insufficient_and_never_calls_rerank(setup, strategy):
    deps, requests, _ = setup([])
    async with deps:
        result = await retrieval.search_knowledge_ch04("型号", strategy=strategy, dependencies=deps)
        assert not result.sufficient and result.hits == [] and result.reason == "no_hits"
        assert requests == []


@pytest.mark.parametrize("response,error", [(httpx.Response(503), httpx.HTTPStatusError),
                                           (httpx.Response(200, json={"results": []}), ValueError)])
async def test_rerank_failure_is_operational_exception_not_low_confidence(setup, response, error):
    deps, _, _ = setup(response=response)
    async with deps:
        with pytest.raises(error):
            await retrieval.search_knowledge_ch04("型号", dependencies=deps)


@pytest.mark.parametrize("component", ["embedder", "store", "model"])
async def test_dependency_failure_propagates(setup, component):
    deps, _, _ = setup()
    def fail(*args, **kwargs):
        raise RuntimeError("dependency unavailable")
    if component == "embedder":
        deps.embedder = SimpleNamespace(embed=fail)
    elif component == "store":
        deps.store = SimpleNamespace(hybrid_search=fail)
    else:
        deps.model = SimpleNamespace(with_structured_output=fail)
    async with deps:
        with pytest.raises(RuntimeError, match="dependency unavailable"):
            await retrieval.search_knowledge_ch04("型号", dependencies=deps)


async def test_documents_are_utf8_bounded_including_total_without_losing_candidates(setup):
    rows = [{**hit(i), "answer": "猫" * 20000} for i in range(50)]
    deps, requests, _ = setup(rows)
    async with deps:
        result = await retrieval.search_knowledge_ch04("型号", dependencies=deps)
        docs = requests[0]["documents"]
        assert len(docs) == 50 and len(result.hits) == 10
        assert all(len(doc.encode("utf-8")) <= 16384 for doc in docs)
        assert sum(len(doc.encode("utf-8")) for doc in docs) <= 262144
        assert all(doc.startswith(f"问题：型号{i}\n答案：") for i, doc in enumerate(docs))
        assert all("�" not in doc for doc in docs)


@pytest.mark.parametrize("count,want", [(1, [0]), (2, [0, 1]), (3, [0, 2, 1]),
                                        (5, [0, 2, 4, 3, 1]), (6, [0, 2, 4, 5, 3, 1])])
async def test_lost_middle_order_retains_each_hit_once(setup, count, want):
    deps, _, _ = setup([hit(i) for i in range(12)])
    async with deps:
        result = await retrieval.search_knowledge_ch04("型号", strategy="hybrid", top_k=count, dependencies=deps)
        assert [x["id"] for x in result.hits] == want


@pytest.mark.parametrize("strategy", ["dense", "bm25", "hybrid", "hybrid_rerank"])
async def test_original_rank_survives_context_reordering_even_with_equal_scores(setup, strategy):
    response = httpx.Response(200, json={"results": [
        {"index": i, "relevance_score": 0.9} for i in range(6)
    ]})
    deps, _, _ = setup([hit(i, score=0.5) for i in range(6)], response)
    async with deps:
        result = await retrieval.search_knowledge_ch04("型号", strategy=strategy, top_k=6, dependencies=deps)
        assert [row["rank"] for row in result.hits] == [1, 3, 5, 6, 4, 2]
        assert [row["id"] for row in sorted(result.hits, key=lambda row: row["rank"])] == [0, 1, 2, 3, 4, 5]


async def test_evaluation_can_reuse_one_validated_understanding_for_fair_strategy_comparison(setup):
    from app.knowledge.query_understanding import QueryUnderstanding
    deps, _, model_calls = setup()
    prepared = QueryUnderstanding(normalized_query="固定问题", search_query="固定问题 同义词", category=None)
    async with deps:
        result = await retrieval.search_knowledge_ch04("原问题", strategy="bm25", dependencies=deps,
                                                       understood_query=prepared)
    assert model_calls == []
    assert result.understood_query == prepared
    assert deps.store.calls[0][1] == "固定问题 同义词"


@pytest.mark.parametrize("kwargs", [{"strategy": "bad"}, {"top_k": 0}, {"top_k": True}, {"top_k": 11}])
async def test_invalid_selection_rejected_before_remote_calls(setup, kwargs):
    deps, requests, model_calls = setup()
    async with deps:
        with pytest.raises(ValueError):
            await retrieval.search_knowledge_ch04("型号", dependencies=deps, **kwargs)
        assert requests == [] and model_calls == []


async def test_injected_dependencies_are_reusable_until_owner_closes(setup):
    deps, requests, _ = setup()
    async with deps:
        await retrieval.search_knowledge_ch04("型号", dependencies=deps)
        await retrieval.search_knowledge_ch04("型号", dependencies=deps)
        client = deps.reranker._client
        assert not client.is_closed and len(requests) == 2
    assert client.is_closed


async def test_default_dependencies_are_closed_even_on_failure(setup, monkeypatch):
    deps, _, _ = setup(response=httpx.Response(503))
    monkeypatch.setattr(retrieval.RetrievalDependencies, "from_settings", classmethod(lambda cls: deps))
    with pytest.raises(httpx.HTTPStatusError):
        await retrieval.search_knowledge_ch04("型号")
    assert deps.reranker._client.is_closed
