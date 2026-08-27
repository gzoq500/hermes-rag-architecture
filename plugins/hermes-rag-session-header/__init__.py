"""Inject the active Hermes session ID into provider request headers."""

from __future__ import annotations

from typing import Any


def add_session_header(**kwargs: Any) -> dict[str, Any] | None:
    session_id = kwargs.get("session_id")
    request = kwargs.get("request")
    if not isinstance(session_id, str) or not session_id or not isinstance(request, dict):
        return None
    updated = dict(request)
    headers = dict(updated.get("extra_headers") or {})
    headers["X-Hermes-Session-Id"] = session_id
    updated["extra_headers"] = headers
    return {
        "request": updated,
        "source": "hermes-rag-session-header",
        "reason": "attached active Hermes session ID",
    }


def register(ctx: Any) -> None:
    ctx.register_middleware("llm_request", add_session_header)
