from rag_gateway.context import (
    build_recalled_message,
    extract_text,
    reduce_context,
)


def test_extract_text_handles_openai_structured_content():
    content = [
        {"type": "text", "text": "first"},
        {"type": "input_text", "text": "second"},
        {"type": "image_url", "image_url": {"url": "https://example.invalid/image"}},
        "third",
    ]

    assert extract_text(content) == "first\nsecond\nthird"


def test_reduce_context_preserves_instructions_and_bounded_recent_suffix():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "developer", "content": "developer"},
        {"role": "user", "content": "recent user"},
        {"role": "assistant", "content": "recent assistant"},
        {"role": "user", "content": "current user"},
    ]

    forwarded, archived = reduce_context(messages, recent_limit=2)

    assert forwarded == [messages[0], messages[3], messages[5], messages[6]]
    assert archived == [messages[1], messages[2], messages[4]]
    assert len(forwarded) < len(messages)


def test_reduce_context_keeps_assistant_tool_call_with_split_tool_suffix():
    messages = [
        {"role": "user", "content": "old"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call-a", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                {"id": "call-b", "type": "function", "function": {"name": "b", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call-a", "content": "a result"},
        {"role": "tool", "tool_call_id": "call-b", "content": "b result"},
        {"role": "assistant", "content": "summary"},
        {"role": "user", "content": "next"},
    ]

    forwarded, archived = reduce_context(messages, recent_limit=4)

    assert forwarded == messages[1:]
    assert archived == [messages[0]]


def test_reduce_context_moves_boundary_before_entire_tool_result_group():
    messages = [
        {"role": "user", "content": "old"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "b", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "a", "content": "a"},
        {"role": "tool", "tool_call_id": "b", "content": "b"},
        {"role": "user", "content": "next"},
    ]

    forwarded, archived = reduce_context(messages, recent_limit=2)

    assert forwarded == messages[1:]
    assert archived == [messages[0]]


def test_reduce_context_always_keeps_the_latest_user_message():
    """A window that would start mid-turn must be expanded to include the user turn.

    Strict OpenAI-compatible providers (CodeBuddy CN) reject requests whose
    conversation opens with an assistant/tool sequence instead of a user
    message (HTTP 400 code 11133). The reduction boundary must therefore
    never drop the last user message of the conversation.
    """
    messages = [
        {"role": "user", "content": "start of current turn"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "result one"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c2", "type": "function", "function": {"name": "t", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "c2", "content": "result two"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c3", "type": "function", "function": {"name": "t", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "c3", "content": "result three"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c4", "type": "function", "function": {"name": "t", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "c4", "content": "result four"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c5", "type": "function", "function": {"name": "t", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "c5", "content": "result five"},
    ]

    forwarded, archived = reduce_context(messages, recent_limit=4)

    assert forwarded[0]["role"] == "user"
    assert forwarded[0]["content"] == "start of current turn"
    assert forwarded == messages
    assert archived == []


def test_reduce_context_keeps_latest_user_when_boundary_lands_mid_turn():
    """Even when older history exists, the boundary moves left to the last user turn."""
    messages = [
        {"role": "user", "content": "old turn"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "current turn"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "k1", "type": "function", "function": {"name": "t", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "k1", "content": "result"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "k2", "type": "function", "function": {"name": "t", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "k2", "content": "result two"},
    ]

    forwarded, archived = reduce_context(messages, recent_limit=3)

    assert forwarded[0]["role"] == "user"
    assert forwarded[0]["content"] == "current turn"
    assert archived == [messages[0], messages[1]]


def test_recalled_context_contains_real_newlines_and_is_a_developer_message():
    recalled = build_recalled_message(["USER: remembered", "TOOL: result"])

    assert recalled["role"] == "developer"
    assert "\\n" not in recalled["content"]
    assert recalled["content"].splitlines() == [
        "[ARCHIVED MEMORY RECALLED]",
        "- USER: remembered",
        "- TOOL: result",
        "[END OF ARCHIVED MEMORY]",
    ]
