"""Explicit Ch04 dense/BM25 read model, independent of Ch03 knowledge."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker

from app.knowledge.embedding import VECTOR_DIMENSION, validate_vectors

COLLECTION_NAME = "knowledge_ch04"
TEXT_LIMITS = {
    "text": 65535, "question": 8192, "answer": 49152,
    "category": 1024, "section_path": 2048, "content_type": 128,
}
OUTPUT_FIELDS = [*TEXT_LIMITS, "is_key_clause"]


@dataclass(frozen=True)
class SearchFilter:
    category: str | None = None
    content_type: str | None = None
    is_key_clause: bool | None = None

    def __post_init__(self):
        for name in ("category", "content_type"):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} filter must be a string")
        if self.is_key_clause is not None and type(self.is_key_clause) is not bool:
            raise TypeError("is_key_clause filter must be a bool")


def build_filter(*, category: str | None = None, content_type: str | None = None,
                 is_key_clause: bool | None = None) -> SearchFilter:
    """Only these typed values can become a Milvus expression."""
    return SearchFilter(category, content_type, is_key_clause)


def _expression(value: SearchFilter | None) -> str:
    if value is None:
        return ""
    if type(value) is not SearchFilter:
        raise TypeError("filter_expr must come from build_filter; raw expressions are forbidden")
    parts = []
    for name in ("category", "content_type", "is_key_clause"):
        item = getattr(value, name)
        if item is not None:
            parts.append(f"{name} == {json.dumps(item, ensure_ascii=False)}")
    return " and ".join(parts)


def validate_text_fields(row: dict) -> None:
    for name, cap in TEXT_LIMITS.items():
        value = row.get(name)
        if not isinstance(value, str):
            raise ValueError(f"SQL chunk {row.get('id')}: {name} must be a string")
        if len(value.encode("utf-8")) > cap:
            raise ValueError(f"SQL chunk {row.get('id')}: {name} exceeds {cap} UTF-8 bytes")


class HybridMilvusStore:
    def __init__(self, *, uri: str | None = None, client: Any | None = None,
                 collection_name: str = COLLECTION_NAME, timeout: float = 20.0):
        self.collection_name = collection_name
        self.timeout = timeout
        if client is None:
            if not uri:
                raise ValueError("Milvus URI is required")
            client = MilvusClient(uri=uri, timeout=timeout)
        self.client = client

    def ensure_collection(self) -> None:
        if self.client.has_collection(self.collection_name):
            self._verify_schema(self.client.describe_collection(self.collection_name))
            self._verify_indexes()
        else:
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("dense", DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION)
            schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
            for name, cap in TEXT_LIMITS.items():
                options = {"enable_analyzer": True, "analyzer_params": {"type": "chinese"}} if name == "text" else {}
                schema.add_field(name, DataType.VARCHAR, max_length=cap, **options)
            schema.add_field("is_key_clause", DataType.BOOL)
            schema.add_function(Function(name="text_bm25", function_type=FunctionType.BM25,
                                         input_field_names=["text"], output_field_names=["sparse"]))
            indexes = self.client.prepare_index_params()
            indexes.add_index("dense", index_type="AUTOINDEX", metric_type="COSINE")
            indexes.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
            self.client.create_collection(collection_name=self.collection_name, schema=schema,
                                          index_params=indexes, consistency_level="Strong")
        self.client.load_collection(self.collection_name)

    def upsert(self, rows: list[dict]) -> None:
        for row in rows:
            validate_text_fields(row)
            if type(row.get("id")) is not int or not -(2**63) <= row["id"] < 2**63:
                raise ValueError(f"SQL chunk {row.get('id')}: id must be INT64")
            if type(row.get("is_key_clause")) is not bool:
                raise ValueError(f"SQL chunk {row['id']}: is_key_clause must be bool")
            if set(row) != {"id", "dense", *OUTPUT_FIELDS}:
                raise ValueError(f"SQL chunk {row['id']}: unexpected fields (sparse is server-generated)")
            validate_vectors([row["dense"]], 1)
        if rows:
            self.client.upsert(collection_name=self.collection_name, data=rows)

    @staticmethod
    def _positive(limit: int) -> None:
        if type(limit) is not int or limit < 1:
            raise ValueError("search limit must be a positive integer")

    def dense_search(self, vector, limit: int = 50, filter_expr: SearchFilter | None = None):
        validate_vectors([vector], 1)
        return self._search(vector, "dense", "COSINE", limit, filter_expr)

    def bm25_search(self, text: str, limit: int = 50, filter_expr: SearchFilter | None = None):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("BM25 query must be nonempty text")
        return self._search(text, "sparse", "BM25", limit, filter_expr)

    def _search(self, data, field, metric, limit, filter_expr):
        self._positive(limit)
        return self._hits(self.client.search(
            collection_name=self.collection_name, data=[data], anns_field=field,
            search_params={"metric_type": metric}, limit=limit,
            filter=_expression(filter_expr), output_fields=OUTPUT_FIELDS, consistency_level="Strong", timeout=self.timeout,
        ))

    def hybrid_search(self, vector, text: str, recall_limit: int = 50, output_limit: int = 10,
                      filter_expr: SearchFilter | None = None):
        self._positive(recall_limit)
        self._positive(output_limit)
        validate_vectors([vector], 1)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("BM25 query must be nonempty text")
        expr = _expression(filter_expr)
        reqs = [AnnSearchRequest(data=[data], anns_field=field, param={"metric_type": metric},
                                 limit=recall_limit, expr=expr)
                for data, field, metric in [(vector, "dense", "COSINE"), (text, "sparse", "BM25")]]
        return self._hits(self.client.hybrid_search(
            collection_name=self.collection_name, reqs=reqs, ranker=RRFRanker(), limit=output_limit,
            output_fields=OUTPUT_FIELDS, consistency_level="Strong", timeout=self.timeout,
        ))

    @staticmethod
    def _hits(batches):
        if not batches:
            return []
        return [{"id": hit["id"], "score": hit["distance"],
                 **{name: hit["entity"][name] for name in OUTPUT_FIELDS}} for hit in batches[0]]

    def count(self) -> int:
        rows = self.client.query(collection_name=self.collection_name, filter="", output_fields=["count(*)"],
                                 consistency_level="Strong")
        return int(rows[0]["count(*)"])

    def ids(self) -> set[int]:
        iterator = self.client.query_iterator(collection_name=self.collection_name, filter="", output_fields=["id"],
                                              batch_size=1000, consistency_level="Strong")
        result = set()
        try:
            while rows := iterator.next():
                result.update(int(row["id"]) for row in rows)
        finally:
            iterator.close()
        return result

    def _verify_schema(self, description):
        def require(condition):
            if not condition:
                raise ValueError("schema contract mismatch")

        try:
            fields = {f["name"]: f for f in description["fields"]}
            expected = {"id": DataType.INT64, "dense": DataType.FLOAT_VECTOR,
                        "sparse": DataType.SPARSE_FLOAT_VECTOR, "is_key_clause": DataType.BOOL,
                        **{name: DataType.VARCHAR for name in TEXT_LIMITS}}
            require(set(fields) == set(expected))
            require(not description.get("enable_dynamic_field", False))
            require(not description.get("auto_id", False))
            require(fields["id"].get("is_primary") and not fields["id"].get("auto_id", False))
            for name, dtype in expected.items():
                require(fields[name]["type"] == dtype)
            require(int(fields["dense"]["params"]["dim"]) == VECTOR_DIMENSION)
            for name, cap in TEXT_LIMITS.items():
                require(int(fields[name]["params"]["max_length"]) == cap)
            params = fields["text"]["params"]
            require(params.get("enable_analyzer") in (True, "true"))
            analyzer = params.get("analyzer_params")
            if isinstance(analyzer, str):
                analyzer = json.loads(analyzer)
            require(analyzer == {"type": "chinese"})
            functions = description["functions"]
            require(len(functions) == 1)
            fn = functions[0]
            require(int(fn["type"]) == int(FunctionType.BM25))
            require(fn["input_field_names"] == ["text"] and fn["output_field_names"] == ["sparse"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("incompatible existing knowledge_ch04 collection schema") from error

    def _verify_indexes(self):
        for field, metric, index_type in [("dense", "COSINE", "AUTOINDEX"), ("sparse", "BM25", "SPARSE_INVERTED_INDEX")]:
            descriptions = [self.client.describe_index(self.collection_name, name)
                            for name in self.client.list_indexes(self.collection_name, field_name=field)]
            if not any(d.get("field_name") == field and d.get("metric_type") == metric
                       and d.get("index_type") == index_type for d in descriptions):
                raise RuntimeError(f"incompatible existing knowledge_ch04 index: {field}")
