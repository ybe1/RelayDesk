"""Disposable live probe for native Chinese BM25 and dense/RRF hybrid search."""

from __future__ import annotations

import json

from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker

from app.config import Settings, settings
from app.knowledge.embedding import EmbeddingClient


COLLECTION = "smoke_ch04_bm25"
MODEL_QUERY = "青鸾巡航无人机"
DOCUMENTS = [
    {"id": 1, "text": "青鸾巡航无人机适用于长航时巡检。", "category": "目标"},
    {"id": 2, "text": "玄武测绘无人机适用于地形测绘。", "category": "对照"},
]


def main(*, config: Settings = settings, client=None, embedder=None) -> dict:
    if client is None:
        client = MilvusClient(uri=config.milvus_uri)
    if embedder is None:
        embedder = EmbeddingClient(
            base_url=config.embed_base_url, api_key=config.embed_api_key, model=config.embed_model
        )
    if client.has_collection(COLLECTION):
        raise RuntimeError(f"disposable collection {COLLECTION} already exists; inspect before rerunning")

    created = False
    try:
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("text", DataType.VARCHAR, max_length=1024,
                         enable_analyzer=True, analyzer_params={"type": "chinese"})
        schema.add_field("category", DataType.VARCHAR, max_length=64)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(Function(
            name="text_bm25", input_field_names=["text"], output_field_names=["sparse"],
            function_type=FunctionType.BM25,
        ))
        index_params = client.prepare_index_params()
        index_params.add_index("dense", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        client.create_collection(collection_name=COLLECTION, schema=schema, index_params=index_params)
        created = True
        vectors = embedder.embed([doc["text"] for doc in DOCUMENTS] + [MODEL_QUERY])
        client.insert(collection_name=COLLECTION, data=[
            {**doc, "dense": vector} for doc, vector in zip(DOCUMENTS, vectors[:2], strict=True)
        ])
        client.load_collection(COLLECTION)

        bm25 = client.search(
            collection_name=COLLECTION, data=[MODEL_QUERY], anns_field="sparse",
            search_params={"metric_type": "BM25"}, limit=2, output_fields=["text", "category"],
            consistency_level="Strong",
        )
        bm25_model_hit = bool(bm25 and bm25[0] and bm25[0][0]["id"] == 1)

        def hybrid(filter_expr: str | None):
            dense_req = AnnSearchRequest(
                data=[vectors[2]], anns_field="dense", param={"metric_type": "COSINE"},
                limit=2, filter=filter_expr,
            )
            bm25_req = AnnSearchRequest(
                data=[MODEL_QUERY], anns_field="sparse", param={"metric_type": "BM25"},
                limit=2, filter=filter_expr,
            )
            return client.hybrid_search(
                collection_name=COLLECTION, reqs=[dense_req, bm25_req], ranker=RRFRanker(),
                limit=2, output_fields=["text", "category"], consistency_level="Strong",
            )

        hits = hybrid(None)
        filtered = hybrid('category == "对照"')
        hybrid_results = hits[0] if hits else []
        filtered_results = filtered[0] if filtered else []
        filter_excluded_model = bool(filtered_results) and all(hit["id"] != 1 for hit in filtered_results)
        if not bm25_model_hit or not hybrid_results or not filter_excluded_model:
            raise RuntimeError("BM25/hybrid/category smoke failed semantic assertions")
        return {
            "bm25_model_hit": bm25_model_hit,
            "hybrid_result_count": len(hybrid_results),
            "filter_excluded_model": filter_excluded_model,
        }
    finally:
        if created:
            client.drop_collection(COLLECTION)


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False))
