"""Idempotent Milvus collection schema management."""

from __future__ import annotations

from typing import Any


def ensure_collection(
    client: Any,
    collection_name: str,
    dimensions: int,
    reset: bool = False,
    data_type: Any | None = None,
) -> str:
    exists = client.has_collection(collection_name=collection_name)
    if exists and not reset:
        return "unchanged"
    if exists:
        client.drop_collection(collection_name=collection_name)
    if data_type is None:
        from pymilvus import DataType

        data_type = DataType
    varchar = getattr(data_type, "VARCHAR", data_type)
    float_vector = getattr(data_type, "FLOAT_VECTOR", data_type)
    int64 = getattr(data_type, "INT64", data_type)
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(
        field_name="id", datatype=varchar, max_length=64, is_primary=True
    )
    schema.add_field(field_name="vector", datatype=float_vector, dim=dimensions)
    schema.add_field(field_name="text", datatype=varchar, max_length=15000)
    schema.add_field(field_name="session_id", datatype=varchar, max_length=512)
    schema.add_field(field_name="role", datatype=varchar, max_length=32)
    schema.add_field(field_name="timestamp", datatype=int64)
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector", index_type="AUTOINDEX", metric_type="COSINE"
    )
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    return "reset" if exists else "created"
