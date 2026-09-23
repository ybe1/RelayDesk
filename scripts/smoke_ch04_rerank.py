"""Fail-fast live probe for SiliconFlow's rerank contract."""

from __future__ import annotations

import json
import math

import httpx

from app.config import Settings, settings


QUERY = "退货政策是什么？"
DOCUMENTS = ["天气晴朗。", "退货政策规定七天内可以退货。"]


def main(*, config: Settings = settings, transport: httpx.BaseTransport | None = None) -> dict:
    if not config.rerank_api_key:
        raise RuntimeError("RERANK_API_KEY is required for the rerank smoke")

    with httpx.Client(timeout=30.0, transport=transport) as client:
        response = client.post(
            config.rerank_base_url.rstrip("/") + "/rerank",
            headers={"Authorization": f"Bearer {config.rerank_api_key}"},
            json={"model": config.rerank_model, "query": QUERY, "documents": DOCUMENTS, "top_n": 2},
        )
        response.raise_for_status()
        payload = response.json()

    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("rerank response has no results list")
    results = payload["results"]
    if not results or len(results) > 2:
        raise ValueError("rerank response has an invalid result count")
    seen: set[int] = set()
    for item in results:
        if not isinstance(item, dict):
            raise ValueError("rerank result is not an object")
        index = item.get("index")
        score = item.get("relevance_score")
        if type(index) is not int or index < 0 or index >= len(DOCUMENTS) or index in seen:
            raise ValueError("rerank result has an invalid index")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("rerank result has an invalid relevance_score")
        seen.add(index)

    top_index = results[0]["index"]
    if top_index != 1:
        raise RuntimeError("return-policy document was not ranked first")
    return {"result_count": len(results), "top_index": top_index, "return_policy_first": True}


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False))
