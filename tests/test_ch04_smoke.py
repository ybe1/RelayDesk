"""Task 1 probes: configuration and rerank response contract."""

import json
from unittest.mock import Mock

import httpx
import pytest

from app.config import Settings


def _settings(**overrides):
    return Settings(
        _env_file=None,
        chat_model="test-model",
        chat_base_url="https://example.invalid/v1",
        chat_api_key="test-key",
        **overrides,
    )


def test_ch04_settings_have_safe_defaults(monkeypatch):
    for name in (
        "RERANK_BASE_URL", "RERANK_API_KEY", "RERANK_MODEL", "MILVUS_COLLECTION",
        "RECALL_TOP_K", "RERANK_TOP_K", "RERANK_MIN_SCORE",
    ):
        monkeypatch.delenv(name, raising=False)
    config = _settings()
    assert config.rerank_base_url == "https://api.siliconflow.cn/v1"
    assert config.rerank_api_key == ""
    assert config.rerank_model == "BAAI/bge-reranker-v2-m3"
    assert config.milvus_collection == "knowledge_ch04"
    assert config.recall_top_k == 50
    assert config.rerank_top_k == 10
    assert config.rerank_min_score == 0.0


def test_rerank_smoke_sends_documented_payload_and_returns_summary():
    from scripts.smoke_ch04_rerank import main

    def respond(request):
        assert request.url == "https://example.invalid/v1/rerank"
        assert request.headers["authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload == {
            "model": "BAAI/bge-reranker-v2-m3",
            "query": "退货政策是什么？",
            "documents": ["天气晴朗。", "退货政策规定七天内可以退货。"],
            "top_n": 2,
        }
        return httpx.Response(200, json={
            "id": "rerank-test",
            "results": [
                {"index": 1, "document": {"text": "退货政策规定七天内可以退货。"}, "relevance_score": 0.91},
                {"index": 0, "document": {"text": "天气晴朗。"}, "relevance_score": 0.1},
            ],
            "meta": {"tokens": {"input_tokens": 12, "output_tokens": 0}},
        })

    summary = main(config=_settings(rerank_base_url="https://example.invalid/v1/", rerank_api_key="test-key"),
                   transport=httpx.MockTransport(respond))
    assert summary["top_index"] == 1
    assert summary["result_count"] == 2
    assert summary["return_policy_first"] is True


@pytest.mark.parametrize("results", [
    [{"index": 1, "relevance_score": 0.9}],
    [{"index": 1, "relevance_score": 0.9}, {"index": 1, "relevance_score": 0.8}],
    [{"index": 2, "relevance_score": 0.9}],
    [{"index": 1, "relevance_score": float("nan")}],
    [{"index": True, "relevance_score": 0.9}],
])
def test_rerank_smoke_rejects_invalid_results(results):
    from scripts.smoke_ch04_rerank import main

    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={
        "id": "rerank-test", "results": results, "meta": {},
    }))
    with pytest.raises(ValueError):
        main(config=_settings(rerank_api_key="test-key"), transport=transport)


def test_bm25_smoke_uses_chinese_analyzer_hybrid_filter_and_disposes_collection():
    from scripts.smoke_ch04_bm25 import main

    class Schema:
        def __init__(self):
            self.fields = []
            self.functions = []

        def add_field(self, *args, **kwargs):
            self.fields.append((args, kwargs))

        def add_function(self, function):
            self.functions.append(function)

    class IndexParams:
        def __init__(self):
            self.indexes = []

        def add_index(self, *args, **kwargs):
            self.indexes.append((args, kwargs))

    class Client:
        def __init__(self):
            self.schema = Schema()
            self.indexes = IndexParams()
            self.created = []
            self.dropped = []
            self.inserted = []

        def has_collection(self, name):
            return False

        def create_schema(self, **kwargs):
            return self.schema

        def prepare_index_params(self):
            return self.indexes

        def create_collection(self, **kwargs):
            self.created.append(kwargs["collection_name"])

        def insert(self, **kwargs):
            self.inserted.extend(kwargs["data"])

        def load_collection(self, name):
            pass

        def search(self, **kwargs):
            return [[{"id": 1, "entity": {"text": self.inserted[0]["text"]}}]]

        def hybrid_search(self, **kwargs):
            if kwargs["reqs"][0].filter:
                return [[{"id": 2, "entity": {"text": self.inserted[1]["text"]}}]]
            return [[{"id": 1, "entity": {"text": self.inserted[0]["text"]}}]]

        def drop_collection(self, name):
            self.dropped.append(name)

    class Embedder:
        def embed(self, texts):
            return [[0.1] * 1024 for _ in texts]

    client = Client()
    summary = main(config=_settings(), client=client, embedder=Embedder())
    assert summary == {
        "bm25_model_hit": True,
        "hybrid_result_count": 1,
        "filter_excluded_model": True,
    }
    assert client.created == ["smoke_ch04_bm25"]
    assert client.dropped == ["smoke_ch04_bm25"]
    assert client.schema.functions
    assert any(kwargs.get("analyzer_params") == {"type": "chinese"} for _, kwargs in client.schema.fields)
    assert {kwargs.get("metric_type") for _, kwargs in client.indexes.indexes} == {"COSINE", "BM25"}


def test_bm25_smoke_drops_created_collection_after_embedding_error():
    from scripts.smoke_ch04_bm25 import main

    client = Mock()
    client.has_collection.return_value = False
    embedder = Mock()
    embedder.embed.side_effect = RuntimeError("embedding failed")

    with pytest.raises(RuntimeError, match="embedding failed"):
        main(config=_settings(), client=client, embedder=embedder)

    client.create_collection.assert_called_once()
    client.drop_collection.assert_called_once_with("smoke_ch04_bm25")


def test_bm25_smoke_preserves_preexisting_collection():
    from scripts.smoke_ch04_bm25 import main

    client = Mock()
    client.has_collection.return_value = True

    with pytest.raises(RuntimeError, match="already exists"):
        main(config=_settings(), client=client, embedder=Mock())

    client.create_collection.assert_not_called()
    client.drop_collection.assert_not_called()
