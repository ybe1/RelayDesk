"""Production rerank client contract at the HTTP boundary."""

import json
import math

import httpx
import pytest

from app.knowledge.rerank import RerankClient


def _client(handler, **overrides):
    return RerankClient(
        base_url="https://example.test/v1/", api_key="private-key",
        transport=httpx.MockTransport(handler), sleep=lambda _: _no_delay(),
        **overrides,
    )


async def _no_delay():
    return None


def _response(indexes_and_scores):
    body = {"results": [
        {"index": index, "relevance_score": score}
        for index, score in indexes_and_scores
    ]}
    return httpx.Response(200, content=json.dumps(body).encode("utf-8"), headers={"content-type": "application/json"})


@pytest.mark.asyncio
async def test_empty_documents_return_without_network_or_key():
    def forbidden(_):
        pytest.fail("empty request reached network")

    client = RerankClient(base_url="https://example.test/v1", api_key="", transport=httpx.MockTransport(forbidden))
    assert await client.rerank("query", [], 10) == []


@pytest.mark.asyncio
async def test_payload_and_response_keep_provider_ranking_with_original_indexes():
    requests = []

    def handler(request):
        requests.append(request)
        return _response([(2, 0.95), (0, 0.61)])

    client = _client(handler)
    assert await client.rerank("退货政策", ["a", "b", "c"], 2) == [(2, 0.95), (0, 0.61)]
    assert len(requests) == 1
    assert str(requests[0].url) == "https://example.test/v1/rerank"
    assert requests[0].headers["authorization"] == "Bearer private-key"
    assert json.loads(requests[0].content) == {
        "model": "BAAI/bge-reranker-v2-m3", "query": "退货政策",
        "documents": ["a", "b", "c"], "top_n": 2,
    }


@pytest.mark.asyncio
async def test_configured_timeout_is_used_for_request():
    seen = []

    def handler(request):
        seen.append(request.extensions["timeout"])
        return _response([(0, 0.8)])

    client = _client(handler, timeout=7.0)
    await client.rerank("q", ["a"], 1)
    assert seen == [{"connect": 7.0, "read": 7.0, "write": 7.0, "pool": 7.0}]


@pytest.mark.asyncio
async def test_top_ten_results_from_fifty_candidates():
    def handler(request):
        assert json.loads(request.content)["top_n"] == 10
        return _response([(index, 1 - index / 100) for index in range(10)])

    client = _client(handler)
    assert len(await client.rerank("q", [str(index) for index in range(50)], 10)) == 10


@pytest.mark.asyncio
async def test_top_n_is_capped_by_available_documents():
    def handler(request):
        assert json.loads(request.content)["top_n"] == 2
        return _response([(1, 0.9), (0, 0.6)])

    client = _client(handler)
    assert await client.rerank("q", ["a", "b"], 10) == [(1, 0.9), (0, 0.6)]


@pytest.mark.asyncio
@pytest.mark.parametrize("results", [
    [(0, 0.8)],
    [(0, 0.8), (0, 0.7)],
    [(0, 0.8), (2, 0.7)],
    [(0, math.nan), (1, 0.7)],
    [(0, math.inf), (1, 0.7)],
    [(True, 0.8), (1, 0.7)],
    [(0, True), (1, 0.7)],
])
async def test_rejects_incomplete_duplicate_invalid_index_or_nonfinite_score(results):
    client = _client(lambda _: _response(results))
    with pytest.raises(ValueError, match="rerank response"):
        await client.rerank("q", ["a", "b"], 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [{}, {"results": {}}, {"results": ["bad", "bad"]}])
async def test_rejects_malformed_response_shape(body):
    client = _client(lambda _: httpx.Response(200, json=body))
    with pytest.raises(ValueError, match="rerank response"):
        await client.rerank("q", ["a", "b"], 2)


@pytest.mark.asyncio
async def test_missing_api_key_fails_before_network():
    def forbidden(_):
        pytest.fail("missing key reached network")

    client = RerankClient(base_url="https://example.test/v1", api_key="", transport=httpx.MockTransport(forbidden))
    with pytest.raises(RuntimeError, match="RERANK_API_KEY"):
        await client.rerank("q", ["document"], 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("documents, top_n, query", [
    (["a"] * 51, 10, "q"),
    (["a"], 11, "q"),
    (["a"], 0, "q"),
    (["a"], 1, ""),
    (["a"], 1, "中" * 2731),
    (["中" * 5462], 1, "q"),
    (["中" * 5000] * 18, 10, "q"),
])
async def test_rejects_oversized_or_invalid_input_before_network(documents, top_n, query):
    def forbidden(_):
        pytest.fail("invalid request reached network")

    client = _client(forbidden)
    with pytest.raises(ValueError, match="rerank input"):
        await client.rerank(query, documents, top_n)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 503, 504])
async def test_retries_only_transient_statuses_with_bounded_backoff(status):
    attempts = 0
    delays = []

    async def sleep(delay):
        delays.append(delay)

    def handler(_):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(status)
        return _response([(0, 0.8)])

    client = RerankClient(base_url="https://example.test/v1", api_key="key", transport=httpx.MockTransport(handler), sleep=sleep)
    assert await client.rerank("q", ["a"], 1) == [(0, 0.8)]
    assert attempts == 3
    assert delays == [0.25, 0.5]


@pytest.mark.asyncio
async def test_retry_budget_stops_after_three_requests():
    attempts = 0

    def handler(_):
        nonlocal attempts
        attempts += 1
        return httpx.Response(503)

    client = _client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await client.rerank("q", ["a"], 1)
    assert attempts == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 500])
async def test_other_http_statuses_do_not_retry(status):
    attempts = 0

    def handler(_):
        nonlocal attempts
        attempts += 1
        return httpx.Response(status)

    client = _client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await client.rerank("q", ["a"], 1)
    assert attempts == 1


@pytest.mark.asyncio
async def test_transport_failure_does_not_retry():
    attempts = 0

    def handler(_):
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("offline")

    client = _client(handler)
    with pytest.raises(httpx.ConnectError):
        await client.rerank("q", ["a"], 1)
    assert attempts == 1


@pytest.mark.asyncio
async def test_http_failure_does_not_log_documents_or_credentials(caplog):
    client = _client(lambda _: httpx.Response(401, json={"error": "sensitive document"}))
    with pytest.raises(httpx.HTTPStatusError):
        await client.rerank("q", ["sensitive document"], 1)
    assert "sensitive document" not in caplog.text
    assert "private-key" not in caplog.text


@pytest.mark.asyncio
async def test_owned_http_client_reuses_transport_until_explicit_close():
    class CountingTransport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.requests = 0
            self.closes = 0

        async def handle_async_request(self, request):
            self.requests += 1
            return _response([(0, 0.8)])

        async def aclose(self):
            self.closes += 1

    transport = CountingTransport()
    client = RerankClient(base_url="https://example.test/v1", api_key="key", transport=transport)
    assert await client.rerank("first", ["a"], 1) == [(0, 0.8)]
    assert await client.rerank("second", ["b"], 1) == [(0, 0.8)]
    assert transport.requests == 2
    assert transport.closes == 0
    await client.aclose()
    assert transport.closes == 1
    with pytest.raises(RuntimeError, match="closed"):
        await client.rerank("third", ["c"], 1)


@pytest.mark.asyncio
async def test_injected_http_client_remains_caller_owned():
    external = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: _response([(0, 0.8)])))
    try:
        client = RerankClient(base_url="https://example.test/v1", api_key="key", client=external)
        assert await client.rerank("first", ["a"], 1) == [(0, 0.8)]
        assert await client.rerank("second", ["b"], 1) == [(0, 0.8)]
        await client.aclose()
        assert external.is_closed is False
    finally:
        await external.aclose()


@pytest.mark.asyncio
async def test_async_context_manager_closes_owned_client():
    transport = httpx.MockTransport(lambda _: _response([(0, 0.8)]))
    async with RerankClient(base_url="https://example.test/v1", api_key="key", transport=transport) as client:
        assert await client.rerank("q", ["a"], 1) == [(0, 0.8)]
    with pytest.raises(RuntimeError, match="closed"):
        await client.rerank("q", ["a"], 1)


@pytest.mark.parametrize("max_retries", [-1, 6, True, False, 1.5, "2"])
def test_constructor_rejects_invalid_retry_budget(max_retries):
    with pytest.raises(ValueError, match="max_retries.*0..5"):
        RerankClient(base_url="https://example.test/v1", api_key="key", max_retries=max_retries)


def test_constructor_rejects_client_and_transport_together():
    external = httpx.AsyncClient()
    try:
        with pytest.raises(ValueError, match="client.*transport"):
            RerankClient(
                base_url="https://example.test/v1", api_key="key",
                client=external, transport=httpx.MockTransport(lambda _: _response([(0, 0.8)])),
            )
    finally:
        import asyncio
        asyncio.run(external.aclose())


@pytest.mark.asyncio
async def test_zero_retry_budget_sends_one_request():
    attempts = 0

    def handler(_):
        nonlocal attempts
        attempts += 1
        return httpx.Response(503)

    client = _client(handler, max_retries=0)
    with pytest.raises(httpx.HTTPStatusError):
        await client.rerank("q", ["a"], 1)
    assert attempts == 1
