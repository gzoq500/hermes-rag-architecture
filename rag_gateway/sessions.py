"""Hermes SQLite session resolution."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Mapping

logger = logging.getLogger(__name__)


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def lookup_session_id(db_path: str, user_query: str) -> str | None:
    normalized = " ".join(user_query.split()).strip()
    if not normalized:
        logger.info("Skipped SQLite session lookup for empty user content")
        return None
    snippet = normalized[:200]
    pattern = f"%{_escape_like(snippet)}%"
    try:
        with sqlite3.connect(db_path) as connection:
            row = connection.execute(
                """
                SELECT session_id
                FROM messages
                WHERE role = ?
                  AND content LIKE ? ESCAPE '\\'
                ORDER BY id DESC
                LIMIT 1
                """,
                ("user", pattern),
            ).fetchone()
    except sqlite3.Error as exc:
        logger.warning("SQLite session lookup failed: %s", exc.__class__.__name__)
        return None
    if row and row[0]:
        logger.info("Resolved session from SQLite")
        return str(row[0])
    logger.info("No SQLite session match")
    return None


def resolve_session_id(
    headers: Mapping[str, str], payload: Mapping[str, object], user_query: str, db_path: str
) -> str:
    header_session = next(
        (value for key, value in headers.items() if key.lower() == "x-hermes-session-id"),
        None,
    )
    if header_session and header_session.strip():
        logger.info("Resolved session from trusted request header")
        return header_session.strip()
    payload_session = payload.get("user")
    if isinstance(payload_session, str) and payload_session.strip() and payload_session != "default_session":
        logger.info("Resolved session from request payload")
        return payload_session.strip()
    database_session = lookup_session_id(db_path, user_query)
    if database_session:
        return database_session
    logger.warning("Using fallback default session; client did not provide an isolated session ID")
    return "default_session"
