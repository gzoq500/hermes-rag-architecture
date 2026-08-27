"""Executable entry point for the Hermes RAG gateway."""

from __future__ import annotations

import os

import uvicorn
from pymilvus import MilvusClient

from .app import create_app
from .config import ConfigurationError, Settings
from .memory import GeminiEmbedder


def build_app():
    """Build the configured FastAPI application and its external clients."""
    settings = Settings.from_env()
    milvus_client = MilvusClient(uri=settings.zilliz_uri, token=settings.zilliz_token)
    embedder = GeminiEmbedder(
        settings.gemini_api_key,
        settings.embedding_model,
        settings.embedding_timeout_seconds,
    )
    return create_app(settings, milvus_client, embedder)


def main() -> None:
    """Run the gateway using host and port values from the environment."""
    host = os.environ.get("RAG_GATEWAY_HOST", "0.0.0.0")
    try:
        port = int(os.environ.get("RAG_GATEWAY_PORT", "20128"))
    except ValueError as exc:
        raise ConfigurationError("RAG_GATEWAY_PORT must be an integer") from exc
    uvicorn.run(build_app(), host=host, port=port)


if __name__ == "__main__":
    main()
