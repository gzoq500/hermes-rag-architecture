"""Incremental OpenAI-compatible SSE parsing and response metadata."""

from __future__ import annotations

import codecs
import json
from collections.abc import Mapping

RELEVANT_RESPONSE_HEADERS = {
    "content-type",
    "cache-control",
    "x-request-id",
    "request-id",
    "retry-after",
    "openai-processing-ms",
    "openai-version",
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
}


def response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        key.lower(): value
        for key, value in headers.items()
        if key.lower() in RELEVANT_RESPONSE_HEADERS
    }


class SSEAccumulator:
    """Parse SSE incrementally without assuming network chunk boundaries."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._buffer = ""
        self.visible_content = ""
        self.reasoning_content = ""

    def feed(self, chunk: bytes) -> None:
        self._buffer += self._decoder.decode(chunk)
        self._drain_complete_events()

    def finish(self) -> None:
        self._buffer += self._decoder.decode(b"", final=True)
        self._drain_complete_events()
        if self._buffer.strip():
            self._consume_event(self._buffer)
        self._buffer = ""

    def _drain_complete_events(self) -> None:
        normalized = self._buffer.replace("\r\n", "\n").replace("\r", "\n")
        while "\n\n" in normalized:
            event, normalized = normalized.split("\n\n", 1)
            self._consume_event(event)
        self._buffer = normalized

    def _consume_event(self, event: str) -> None:
        data_lines = [line[5:].lstrip() for line in event.split("\n") if line.startswith("data:")]
        if not data_lines:
            return
        data = "".join(data_lines)
        if data == "[DONE]":
            return
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return
        delta = choices[0].get("delta", {}) if isinstance(choices[0], dict) else {}
        if not isinstance(delta, dict):
            return
        content = delta.get("content")
        if isinstance(content, str):
            self.visible_content += content
        for key in ("reasoning_content", "reasoning"):
            reasoning = delta.get(key)
            if isinstance(reasoning, str):
                self.reasoning_content += reasoning

    @property
    def archive_text(self) -> str:
        if self.reasoning_content and self.visible_content:
            return f"REASONING:\n{self.reasoning_content}\n\nANSWER:\n{self.visible_content}"
        return self.visible_content or self.reasoning_content
