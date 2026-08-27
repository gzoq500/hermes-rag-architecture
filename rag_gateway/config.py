"""Environment-backed gateway configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping
import os


class ConfigurationError(RuntimeError):
    """Raised when gateway configuration is missing or invalid."""


def _integer(environment: Mapping[str, str], name: str, default: int) -> int:
    raw = environment.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if value < 0:
        raise ConfigurationError(f"{name} must be zero or greater")
    return value


def _boolean(environment: Mapping[str, str], name: str, default: bool) -> bool:
    raw = environment.get(name, str(default)).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be a boolean")


@dataclass(frozen=True)
class Settings:
    zilliz_uri: str = field(repr=False)
    zilliz_token: str = field(repr=False)
    gemini_api_key: str = field(repr=False)
    collection_name: str = "hermes_gemini_memory"
    db_path: str = "~/.hermes/state.db"
    router_base_url: str = "http://127.0.0.1:20130"
    embedding_model: str = "models/gemini-embedding-2"
    embedding_dimensions: int = 3072
    embedding_timeout_seconds: float = 10.0
    upstream_timeout_seconds: float = 120.0
    recent_messages: int = 24
    retrieval_limit: int = 3
    retrieval_min_chars: int = 10
    chunk_max_chars: int = 8000
    minimum_ingest_chars: int = 15
    max_ingest_records_per_request: int = 32
    reset_collection: bool = False

    @classmethod
    def from_env(cls, environment: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if environment is None else environment
        required = ("ZILLIZ_URI", "ZILLIZ_TOKEN", "GEMINI_API_KEY")
        missing = [name for name in required if not env.get(name, "").strip()]
        if missing:
            raise ConfigurationError(
                "Missing required environment variables: " + ", ".join(missing)
            )
        try:
            embedding_timeout = float(env.get("GEMINI_TIMEOUT_SECONDS", "10"))
            upstream_timeout = float(env.get("UPSTREAM_TIMEOUT_SECONDS", "120"))
        except ValueError as exc:
            raise ConfigurationError("Timeout environment variables must be numbers") from exc
        return cls(
            zilliz_uri=env["ZILLIZ_URI"],
            zilliz_token=env["ZILLIZ_TOKEN"],
            gemini_api_key=env["GEMINI_API_KEY"],
            collection_name=env.get("ZILLIZ_COLLECTION", "hermes_gemini_memory"),
            db_path=os.path.expanduser(env.get("HERMES_DB_PATH", "~/.hermes/state.db")),
            router_base_url=env.get("ROUTER_BASE_URL", "http://127.0.0.1:20130").rstrip("/"),
            embedding_model=env.get("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2"),
            embedding_dimensions=_integer(env, "GEMINI_EMBEDDING_DIMENSIONS", 3072),
            embedding_timeout_seconds=embedding_timeout,
            upstream_timeout_seconds=upstream_timeout,
            recent_messages=_integer(env, "RAG_RECENT_MESSAGES", 24),
            retrieval_limit=_integer(env, "RAG_RETRIEVAL_LIMIT", 3),
            retrieval_min_chars=_integer(env, "RAG_RETRIEVAL_MIN_CHARS", 10),
            chunk_max_chars=_integer(env, "RAG_CHUNK_MAX_CHARS", 8000),
            minimum_ingest_chars=_integer(env, "RAG_MINIMUM_INGEST_CHARS", 15),
            max_ingest_records_per_request=_integer(
                env, "RAG_MAX_INGEST_RECORDS_PER_REQUEST", 32
            ),
            reset_collection=_boolean(env, "RAG_RESET_COLLECTION", False),
        )
