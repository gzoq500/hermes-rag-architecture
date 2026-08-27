"""Message normalization and deterministic context reduction."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

INSTRUCTION_ROLES = {"system", "developer"}
ARCHIVABLE_ROLES = {"user", "assistant", "tool"}


def extract_text(content: Any) -> str:
    """Return visible text from string or OpenAI structured content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in {"text", "input_text", "output_text"}:
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else ""
    return ""


def _tool_call_ids(message: dict[str, Any]) -> set[str]:
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return set()
    return {
        str(call["id"])
        for call in calls
        if isinstance(call, dict) and call.get("id") is not None
    }


def _expand_tool_boundary(messages: list[dict[str, Any]], start: int) -> int:
    """Move a suffix boundary left so tool exchanges remain self-contained."""
    if start >= len(messages) or messages[start].get("role") != "tool":
        return start
    first_tool = start
    while first_tool > 0 and messages[first_tool - 1].get("role") == "tool":
        first_tool -= 1
    tool_ids: set[str] = set()
    index = first_tool
    while index < len(messages) and messages[index].get("role") == "tool":
        call_id = messages[index].get("tool_call_id")
        if call_id is not None:
            tool_ids.add(str(call_id))
        index += 1
    assistant_index = first_tool - 1
    if assistant_index < 0 or messages[assistant_index].get("role") != "assistant":
        return start
    if not tool_ids.issubset(_tool_call_ids(messages[assistant_index])):
        return start
    return assistant_index


def reduce_context(
    messages: list[dict[str, Any]], recent_limit: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep all instructions and a bounded recent conversation suffix.

    The suffix may exceed the configured count only to retain the assistant
    tool-call message required by leading tool results.
    """
    copied = deepcopy(messages)
    instruction_indexes = {
        index for index, message in enumerate(copied) if message.get("role") in INSTRUCTION_ROLES
    }
    conversation_indexes = [index for index in range(len(copied)) if index not in instruction_indexes]
    keep_conversation: set[int] = set()
    if recent_limit > 0 and conversation_indexes:
        suffix_position = max(0, len(conversation_indexes) - recent_limit)
        conversational = [copied[index] for index in conversation_indexes]
        suffix_position = _expand_tool_boundary(conversational, suffix_position)
        keep_conversation.update(conversation_indexes[suffix_position:])
    keep_indexes = instruction_indexes | keep_conversation
    forwarded = [message for index, message in enumerate(copied) if index in keep_indexes]
    archived = [
        message
        for index, message in enumerate(copied)
        if index not in keep_indexes and message.get("role") in ARCHIVABLE_ROLES
    ]
    return forwarded, archived


def build_recalled_message(texts: list[str]) -> dict[str, str]:
    lines = ["[ARCHIVED MEMORY RECALLED]"]
    lines.extend(f"- {text}" for text in texts if text)
    lines.append("[END OF ARCHIVED MEMORY]")
    return {"role": "developer", "content": "\n".join(lines)}
