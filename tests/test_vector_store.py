import pytest


class FakeSchema:
    def __init__(self):
        self.fields = []

    def add_field(self, name, dtype, **kwargs):
        self.fields.append((name, dtype, kwargs))


class FakeIndexParams:
    def __init__(self):
        self.indexes = []

    def add_index(self, field_name, **kwargs):
        self.indexes.append((field_name, kwargs))


class FakeMilvus:
    def __init__(self, *, exists=False, description=None):
        self.exists = exists
        self.description = description
        self.schema = None
        self.index_params = None
        self.upserted = []
        self.search_args = None
        self.loaded = []

    def has_collection(self, collection_name):
        assert collection_name == "knowledge"
        return self.exists

    def create_schema(self, **kwargs):
        self.schema_kwargs = kwargs
        self.schema = FakeSchema()
        return self.schema

    def prepare_index_params(self):
        self.index_params = FakeIndexParams()
        return self.index_params

    def create_collection(self, **kwargs):
        self.create_args = kwargs

    def describe_collection(self, collection_name):
        assert collection_name == "knowledge"
        return self.description

    def load_collection(self, collection_name):
        self.loaded.append(collection_name)

    def upsert(self, **kwargs):
        self.upserted.append(kwargs)

    def search(self, **kwargs):
        self.search_args = kwargs
        return [[
            {"id": 7, "distance": 0.88, "entity": {"question": "退货", "answer": "七天", "category": "售后"}}
        ]]

    def get_collection_stats(self, collection_name):
        assert collection_name == "knowledge"
        return {"row_count": "2"}


def _description(dim=1024, *, answer_max=65535):
    return {"fields": [
        {"name": "id", "type": "Int64", "is_primary": True, "auto_id": False},
        {"name": "vector", "type": "FloatVector", "params": {"dim": dim}},
        {"name": "question", "type": "VarChar", "params": {"max_length": 8192}},
        {"name": "answer", "type": "VarChar", "params": {"max_length": answer_max}},
        {"name": "category", "type": "VarChar", "params": {"max_length": 1020}},
    ]}


def _row(**overrides):
    row = {"id": 3, "vector": [0.0] * 1024, "question": "怎么退货", "answer": "七天无理由", "category": "售后"}
    row.update(overrides)
    return row


def test_ensure_collection_creates_explicit_full_read_model_schema_and_cosine_index():
    from app.knowledge.vector_store import MilvusStore

    client = FakeMilvus()
    store = MilvusStore(client=client)
    store.ensure_collection()

    assert client.schema_kwargs == {"auto_id": False, "enable_dynamic_field": False}
    assert [(name, kwargs) for name, _, kwargs in client.schema.fields] == [
        ("id", {"is_primary": True}),
        ("vector", {"dim": 1024}),
        ("question", {"max_length": 8192}),
        ("answer", {"max_length": 65535}),
        ("category", {"max_length": 1020}),
    ]
    assert client.index_params.indexes == [("vector", {"index_type": "AUTOINDEX", "metric_type": "COSINE"})]
    assert client.create_args["collection_name"] == "knowledge"
    assert client.create_args["schema"] is client.schema
    assert client.create_args["index_params"] is client.index_params
    assert client.loaded == ["knowledge"]


def test_ensure_collection_rejects_incompatible_existing_schema():
    from app.knowledge.vector_store import MilvusStore

    client = FakeMilvus(exists=True, description=_description(dim=768))
    with pytest.raises(RuntimeError, match="incompatible"):
        MilvusStore(client=client).ensure_collection()


def test_upsert_preflights_vector_and_utf8_byte_limits():
    from app.knowledge.vector_store import MilvusStore

    client = FakeMilvus()
    store = MilvusStore(client=client)
    store.upsert([_row()])
    assert client.upserted == [{"collection_name": "knowledge", "data": [_row()]}]

    with pytest.raises(ValueError, match="question"):
        store.upsert([_row(question="你" * 2731)])
    with pytest.raises(ValueError, match="vector"):
        store.upsert([_row(vector=[0.0] * 1023)])


def test_search_returns_all_online_read_fields_and_requests_them_from_milvus():
    from app.knowledge.vector_store import MilvusStore

    client = FakeMilvus()
    hits = MilvusStore(client=client).search([0.0] * 1024, top_k=5)

    assert client.search_args == {
        "collection_name": "knowledge", "data": [[0.0] * 1024,], "limit": 5,
        "output_fields": ["question", "answer", "category"],
    }
    assert hits == [{"id": 7, "score": 0.88, "question": "退货", "answer": "七天", "category": "售后"}]


def test_count_uses_collection_stats_row_count():
    from app.knowledge.vector_store import MilvusStore

    assert MilvusStore(client=FakeMilvus()).count() == 2
