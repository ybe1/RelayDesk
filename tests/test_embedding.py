import json
import math

import httpx
import pytest


def _vector(value: float = 0.0) -> list[float]:
    return [value] * 1024


def test_embed_batches_requests_and_restores_response_index_order():
    from app.knowledge.embedding import EmbeddingClient

    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = json.loads(request.content)
        return httpx.Response(200, json={"data": [
            {"index": 1, "embedding": _vector(2.0)},
            {"index": 0, "embedding": _vector(1.0)},
        ]})

    client = EmbeddingClient(
        base_url="https://example.test/v1", api_key="secret",
        transport=httpx.MockTransport(handler), sleep=lambda _: None,
    )

    vectors = client.embed(["one", "two"])

    assert [vector[0] for vector in vectors] == [1.0, 2.0]
    assert requests[0].url.path == "/v1/embeddings"
    assert requests[0].headers["authorization"] == "Bearer secret"
    assert json.loads(requests[0].content) == {
        "model": "BAAI/bge-m3", "input": ["one", "two"], "encoding_format": "float"
    }


def test_embed_splits_batches_at_sixteen():
    from app.knowledge.embedding import EmbeddingClient

    payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        payloads.append(body)
        return httpx.Response(200, json={"data": [
            {"index": index, "embedding": _vector(float(index))}
            for index, _ in enumerate(body["input"])
        ]})

    client = EmbeddingClient(base_url="https://example.test", api_key="key", transport=httpx.MockTransport(handler))
    vectors = client.embed([str(index) for index in range(17)])

    assert [len(payload["input"]) for payload in payloads] == [16, 1]
    assert len(vectors) == 17


@pytest.mark.parametrize("vectors", [
    [_vector()] * 2,
    [[0.0] * 1023],
    [[math.nan] * 1024],
    [[math.inf] * 1024],
])
def test_validate_vectors_rejects_bad_count_dimension_or_non_finite_values(vectors):
    from app.knowledge.embedding import validate_vectors

    with pytest.raises(ValueError):
        validate_vectors(vectors, expected_count=1)


def test_embed_requires_api_key_only_when_called():
    from app.knowledge.embedding import EmbeddingClient

    client = EmbeddingClient(base_url="https://example.test", api_key="")
    with pytest.raises(RuntimeError, match="EMBED_API_KEY"):
        client.embed(["hello"])


@pytest.mark.parametrize("status", [429, 503, 504])
def test_embed_retries_transient_http_statuses_a_bounded_number_of_times(status):
    from app.knowledge.embedding import EmbeddingClient

    attempts = 0
    sleeps = []

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(status, json={"error": "retry"})
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": _vector()}]})

    client = EmbeddingClient(
        base_url="https://example.test", api_key="key", transport=httpx.MockTransport(handler),
        max_retries=2, sleep=sleeps.append,
    )

    assert client.embed(["hello"]) == [_vector()]
    assert attempts == 3
    assert sleeps == [0.25, 0.5]


def test_embed_does_not_retry_non_transient_http_statuses():
    from app.knowledge.embedding import EmbeddingClient

    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, json={"error": "bad request"})

    client = EmbeddingClient(base_url="https://example.test", api_key="key", transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        client.embed(["hello"])
    assert attempts == 1
