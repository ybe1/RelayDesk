import pytest


class FakeEmbedder:
    def __init__(self, *, vector=None, error=None):
        self.vector = vector or [0.25, 0.75]
        self.error = error
        self.queries = []

    def embed(self, queries):
        self.queries.append(queries)
        if self.error:
            raise self.error
        return [self.vector]


class FakeStore:
    def __init__(self, hits=None, *, error=None):
        self.hits = hits or []
        self.error = error
        self.searches = []

    def search(self, vector, top_k):
        self.searches.append((vector, top_k))
        if self.error:
            raise self.error
        return self.hits


def test_search_knowledge_embeds_then_returns_ranked_hits_above_minimum_score():
    from app.knowledge.retrieval import search_knowledge

    embedder = FakeEmbedder()
    store = FakeStore([
        {"id": 2, "score": 0.72, "question": "退货条件", "answer": "七天内申请", "category": "售后"},
        {"id": 3, "score": 0.40, "question": "猫粮保存", "answer": "阴凉处保存", "category": "商品"},
        {"id": 1, "score": 0.83, "question": "运费怎么算", "answer": "满99元包邮", "category": "配送"},
    ])

    hits = search_knowledge("邮费是多少", embedder, store)

    assert embedder.queries == [["邮费是多少"]]
    assert store.searches == [([0.25, 0.75], 5)]
    assert hits == [
        {"id": 1, "score": 0.83, "question": "运费怎么算", "answer": "满99元包邮", "category": "配送"},
        {"id": 2, "score": 0.72, "question": "退货条件", "answer": "七天内申请", "category": "售后"},
    ]


def test_search_knowledge_returns_empty_when_milvus_has_no_hits():
    from app.knowledge.retrieval import search_knowledge

    assert search_knowledge("邮费是多少", FakeEmbedder(), FakeStore()) == []


@pytest.mark.parametrize("embed_error, store_error", [
    (RuntimeError("embedding unavailable"), None),
    (None, RuntimeError("milvus unavailable")),
])
def test_search_knowledge_surfaces_embedding_or_milvus_errors(embed_error, store_error):
    from app.knowledge.retrieval import search_knowledge

    with pytest.raises(RuntimeError, match="unavailable"):
        search_knowledge("邮费是多少", FakeEmbedder(error=embed_error), FakeStore(error=store_error))
