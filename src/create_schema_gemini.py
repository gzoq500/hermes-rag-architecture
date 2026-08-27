#!/usr/bin/env python3
"""Create or explicitly reset the configured Milvus collection."""

from pymilvus import MilvusClient

from rag_gateway.config import Settings
from rag_gateway.schema import ensure_collection


def main() -> None:
    settings = Settings.from_env()
    client = MilvusClient(uri=settings.zilliz_uri, token=settings.zilliz_token)
    result = ensure_collection(
        client,
        settings.collection_name,
        settings.embedding_dimensions,
        reset=settings.reset_collection,
    )
    print(f"Collection {settings.collection_name}: {result}")


if __name__ == "__main__":
    main()
