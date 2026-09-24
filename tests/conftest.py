"""Local HTTP transport for structured-output tests; no external requests."""

import json

import httpx
import pytest
from langchain_openai import ChatOpenAI


@pytest.fixture
async def structured_transport():
    clients = []

    def make(tool_name, output):
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "local", "object": "chat.completion", "created": 0, "model": "local",
                "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": None, "tool_calls": [{
                        "id": "call_local", "type": "function", "function": {
                            "name": tool_name, "arguments": json.dumps(output, ensure_ascii=False),
                        },
                    }],
                }}],
            })

        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        clients.append(client)
        model = ChatOpenAI(model="local", api_key="local-test-key", base_url="https://local.invalid/v1",
                           http_async_client=client, max_retries=0)
        return model, requests

    yield make
    for client in clients:
        await client.aclose()
