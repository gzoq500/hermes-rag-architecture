"""Embedding and vector-memory operations."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import logging
import threading
import time
from typing import Any, Protocol

import httpx

from .context import extract_text

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float] | None: ...


def chunk_text(text: str, max_chars: int) -> list[str]:
    if not text:
        return []
    return [text[index : index + max_chars] for index in range(0, len(text), max_chars)]


def deterministic_record_id(session_id: str, role: str, text: str, chunk_index: int) -> str:
    canonical = json.dumps(
        [session_id, role, text, chunk_index], ensure_ascii=False, separators=(",", ":")
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def escape_milvus_string(value: str) -> str:
    """Escape a value for a Milvus double-quoted string literal."""
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


class GeminiEmbedder:
    def __init__(
        self,
        api_keys: list[str],
        model: str,
        timeout: float,
        *,
        cooldown_seconds: float = 60.0,
    ):
        if not api_keys:
            raise ValueError("At least one API key is required")
        self._api_keys = api_keys
        self._current_idx = 0
        self._model = model
        self._timeout = timeout
        self._cooldown_seconds = cooldown_seconds
        self._cooldown_until = [0.0 for _ in api_keys]
        self._state_lock = threading.Lock()
        self._last_exhausted_log_at = 0.0

    def embed(self, text: str) -> list[float] | None:
        url = f"https://generativelanguage.googleapis.com/v1beta/{self._model}:embedContent"
        now = time.monotonic()
        with self._state_lock:
            indexes = [
                (self._current_idx + offset) % len(self._api_keys)
                for offset in range(len(self._api_keys))
                if self._cooldown_until[
                    (self._current_idx + offset) % len(self._api_keys)
                ] <= now
            ]

        if not indexes:
            return None

        for index in indexes:
            key = self._api_keys[index]
            try:
                response = httpx.post(
                    url,
                    params={"key": key},
                    json={"model": self._model, "content": {"parts": [{"text": text}]}},
                    timeout=self._timeout,
                )
                if response.status_code == 429:
                    with self._state_lock:
                        self._cooldown_until[index] = (
                            time.monotonic() + self._cooldown_seconds
                        )
                        self._current_idx = (index + 1) % len(self._api_keys)
                    logger.warning(
                        "ECC/Failover: Gemini key index %d rate limited; "
                        "cooling down for %.0fs",
                        index,
                        self._cooldown_seconds,
                    )
                    continue
                response.raise_for_status()
                values = response.json()["embedding"]["values"]
                with self._state_lock:
                    self._cooldown_until[index] = 0.0
                    self._current_idx = (index + 1) % len(self._api_keys)
                return [float(value) for value in values]
            except httpx.HTTPStatusError as exc:
                logger.warning("Embedding request failed: %s", exc.__class__.__name__)
                return None
            except Exception as exc:
                logger.warning("Embedding request failed: %s", exc.__class__.__name__)
                return None
        now = time.monotonic()
        with self._state_lock:
            should_log = now - self._last_exhausted_log_at >= self._cooldown_seconds
            if should_log:
                self._last_exhausted_log_at = now
        if should_log:
            logger.warning(
                "ECC/Failover: All Gemini API keys are cooling down after rate limits"
            )
        return None


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    text: str
    session_id: str
    role: str
    timestamp: int

    @classmethod
    def from_message(
        cls,
        session_id: str,
        message: dict[str, Any],
        timestamp: int,
        chunk_max_chars: int,
    ) -> list["MemoryRecord"]:
        role = str(message.get("role", ""))
        if role not in {"user", "assistant", "tool"}:
            return []
        visible = extract_text(message.get("content"))
        if not visible and role == "assistant" and message.get("tool_calls"):
            visible = json.dumps(message["tool_calls"], ensure_ascii=False, sort_keys=True)
        if not visible:
            return []
        prefix = role.upper()
        if role == "tool" and message.get("tool_call_id"):
            prefix += f"[{message['tool_call_id']}]"
        chunks = chunk_text(visible, chunk_max_chars)
        return [
            cls(
                id=deterministic_record_id(session_id, role, chunk, index),
                text=f"{prefix}: {chunk}",
                session_id=session_id,
                role=role,
                timestamp=timestamp,
            )
            for index, chunk in enumerate(chunks)
        ]


def _existing_record_ids(client: Any, collection_name: str, ids: list[str]) -> set[str]:
    if not ids:
        return set()
    expression = "id in [" + ",".join(f'\"{record_id}\"' for record_id in ids) + "]"
    try:
        rows = client.query(
            collection_name=collection_name,
            filter=expression,
            output_fields=["id"],
        )
    except Exception as exc:
        logger.warning("Existing memory ID lookup failed: %s", exc.__class__.__name__)
        return set()
    return {
        str(row["id"])
        for row in rows
        if isinstance(row, dict) and row.get("id") is not None
    }


def archive_messages(
    client: Any,
    embedder: EmbeddingProvider,
    session_id: str,
    messages: list[dict[str, Any]],
    timestamp: int,
    minimum_chars: int,
    *,
    collection_name: str = "hermes_gemini_memory",
    chunk_max_chars: int = 8000,
    max_records: int = 32,
) -> int:
    raw_candidates = [
        record
        for message in messages
        for record in MemoryRecord.from_message(
            session_id, message, timestamp, chunk_max_chars
        )
        if len(record.text) >= minimum_chars
    ]
    candidates = list({record.id: record for record in raw_candidates}.values())
    existing = _existing_record_ids(
        client, collection_name, [record.id for record in candidates]
    )
    pending = [record for record in candidates if record.id not in existing][:max_records]
    documents: list[dict[str, Any]] = []
    for record in pending:
        vector = embedder.embed(record.text)
        if vector:
            documents.append({**record.__dict__, "vector": vector})
    if documents:
        client.upsert(collection_name=collection_name, data=documents)
        logger.info("Upserted %d memory records", len(documents))
    return len(documents)


def search_memory(
    client: Any,
    embedder: EmbeddingProvider,
    collection_name: str,
    query: str,
    session_id: str | None,
    limit: int,
) -> list[str]:
    vector = embedder.embed(query)
    if not vector:
        return []
    filter_expression = ""
    if session_id:
        filter_expression = f'session_id == "{escape_milvus_string(session_id)}"'
    try:
        results = client.search(
            collection_name=collection_name,
            data=[vector],
            limit=limit,
            filter=filter_expression,
            output_fields=["text"],
        )
    except Exception as exc:
        logger.warning("Memory search failed: %s", exc.__class__.__name__)
        return []
    if not results:
        return []
    return [
        str(hit["entity"]["text"])
        for hit in results[0]
        if isinstance(hit, dict)
        and isinstance(hit.get("entity"), dict)
        and hit["entity"].get("text")
    ]
