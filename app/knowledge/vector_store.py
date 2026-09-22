"""Milvus full-field read model for Ch03 online retrieval."""

from __future__ import annotations

import math
from typing import Any

from pymilvus import DataType, MilvusClient

from app.knowledge.embedding import VECTOR_DIMENSION


COLLECTION_NAME = "knowledge"
QUESTION_MAX_BYTES = 8192
ANSWER_MAX_BYTES = 65535
CATEGORY_MAX_BYTES = 1020
_TEXT_LIMITS = {
    "question": QUESTION_MAX_BYTES,
    "answer": ANSWER_MAX_BYTES,
    "category": CATEGORY_MAX_BYTES,
}


class MilvusStore:
    """Own the explicit Milvus collection schema and online read mapping."""

    def __init__(self, *, uri: str | None = None, client: Any | None = None) -> None:
        if client is None:
            if not uri:
                raise ValueError("Milvus URI is required when no client is supplied")
            client = MilvusClient(uri=uri)
        self.client = client

    def ensure_collection(self) -> None:
        if self.client.has_collection(COLLECTION_NAME):
            self._verify_schema(self.client.describe_collection(COLLECTION_NAME))
            self._verify_vector_index()
            self.client.load_collection(COLLECTION_NAME)
            return

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION)
        schema.add_field("question", DataType.VARCHAR, max_length=QUESTION_MAX_BYTES)
        schema.add_field("answer", DataType.VARCHAR, max_length=ANSWER_MAX_BYTES)
        schema.add_field("category", DataType.VARCHAR, max_length=CATEGORY_MAX_BYTES)
        index_params = self.client.prepare_index_params()
        index_params.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
        )
        self.client.load_collection(COLLECTION_NAME)

    def upsert(self, rows: list[dict]) -> None:
        for row in rows:
            self._validate_row(row)
        if rows:
            self.client.upsert(collection_name=COLLECTION_NAME, data=rows)

    def search(self, vector: list[float], top_k: int) -> list[dict]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self._validate_vector(vector)
        batches = self.client.search(
            collection_name=COLLECTION_NAME,
            data=[vector],
            limit=top_k,
            output_fields=["question", "answer", "category"],
        )
        if not batches:
            return []
        return [self._hit_to_dict(hit) for hit in batches[0]]

    def count(self) -> int:
        return int(self.client.get_collection_stats(COLLECTION_NAME).get("row_count", 0))

    def _validate_row(self, row: dict) -> None:
        required = {"id", "vector", "question", "answer", "category"}
        missing = required.difference(row)
        if missing:
            raise ValueError(f"vector row is missing required fields: {', '.join(sorted(missing))}")
        if not isinstance(row["id"], int) or isinstance(row["id"], bool):
            raise ValueError("id must be an INT64 integer")
        self._validate_vector(row["vector"])
        for field, maximum in _TEXT_LIMITS.items():
            value = row[field]
            if not isinstance(value, str):
                raise ValueError(f"{field} must be a string")
            byte_length = len(value.encode("utf-8"))
            if byte_length > maximum:
                raise ValueError(f"{field} exceeds the Milvus VARCHAR limit of {maximum} UTF-8 bytes")

    @staticmethod
    def _validate_vector(vector: object) -> None:
        if not isinstance(vector, list) or len(vector) != VECTOR_DIMENSION:
            raise ValueError(f"vector must contain exactly {VECTOR_DIMENSION} values")
        try:
            finite = all(math.isfinite(float(value)) for value in vector)
        except (TypeError, ValueError):
            finite = False
        if not finite:
            raise ValueError("vector must contain only finite numeric values")

    @staticmethod
    def _hit_to_dict(hit: dict) -> dict:
        entity = hit.get("entity", hit)
        return {
            "id": hit.get("id", hit.get("pk")),
            "score": hit.get("distance", hit.get("score")),
            "question": entity["question"],
            "answer": entity["answer"],
            "category": entity["category"],
        }

    def _verify_schema(self, description: object) -> None:
        if not isinstance(description, dict) or not isinstance(description.get("fields"), list):
            raise RuntimeError("incompatible existing knowledge collection schema")
        fields = {field.get("name"): field for field in description["fields"] if isinstance(field, dict)}
        required = {
            "id": ("int64", None),
            "vector": ("floatvector", VECTOR_DIMENSION),
            "question": ("varchar", QUESTION_MAX_BYTES),
            "answer": ("varchar", ANSWER_MAX_BYTES),
            "category": ("varchar", CATEGORY_MAX_BYTES),
        }
        for name, (expected_type, expected_limit) in required.items():
            field = fields.get(name)
            if field is None or self._normal_type(field.get("type")) != expected_type:
                raise RuntimeError(f"incompatible existing knowledge collection schema: {name}")
            if expected_limit is not None and int(field.get("params", {}).get("dim", field.get("params", {}).get("max_length", -1))) != expected_limit:
                raise RuntimeError(f"incompatible existing knowledge collection schema: {name}")
        identifier = fields["id"]
        if not identifier.get("is_primary") or identifier.get("auto_id") is not False:
            raise RuntimeError("incompatible existing knowledge collection schema: id primary key")

    def _verify_vector_index(self) -> None:
        index_names = self.client.list_indexes(COLLECTION_NAME, field_name="vector")
        if not index_names:
            raise RuntimeError("incompatible existing knowledge collection index: vector COSINE index is missing")
        for index_name in index_names:
            description = self.client.describe_index(COLLECTION_NAME, index_name)
            if (
                isinstance(description, dict)
                and description.get("field_name") == "vector"
                and str(description.get("metric_type", "")).upper() == "COSINE"
            ):
                return
        raise RuntimeError("incompatible existing knowledge collection index: vector must use COSINE")

    @staticmethod
    def _normal_type(value: object) -> str:
        known_types = {
            int(DataType.INT64): "int64",
            int(DataType.FLOAT_VECTOR): "floatvector",
            int(DataType.VARCHAR): "varchar",
        }
        if isinstance(value, int) and value in known_types:
            return known_types[value]
        return "".join(character for character in str(value).lower() if character.isalnum())
