from app.config import settings
from app.db.session import SessionLocal
from app.knowledge.embedding import EmbeddingClient
from app.knowledge.hybrid_store import HybridMilvusStore
from app.knowledge.incremental_indexer import index_pending_review_chunks


def main() -> int:
    embedder = EmbeddingClient(base_url=settings.embed_base_url, api_key=settings.embed_api_key,
                               model=settings.embed_model)
    store = HybridMilvusStore(uri=settings.milvus_uri, collection_name=settings.milvus_collection)
    store.ensure_collection()
    print(index_pending_review_chunks(SessionLocal, embedder, store))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
