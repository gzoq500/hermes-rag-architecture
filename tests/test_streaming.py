import json

from rag_gateway.streaming import SSEAccumulator, response_headers


def sse(payload):
    return f"data: {json.dumps(payload)}\n\n".encode()


def test_sse_parser_handles_arbitrarily_split_utf8_and_event_boundaries():
    raw = (
        sse({"choices": [{"delta": {"content": "hé"}}]})
        + sse({"choices": [{"delta": {"reasoning_content": "think"}}]})
        + b"data: [DONE]\n\n"
    )
    accumulator = SSEAccumulator()

    for byte in raw:
        accumulator.feed(bytes([byte]))
    accumulator.finish()

    assert accumulator.visible_content == "hé"
    assert accumulator.reasoning_content == "think"
    assert accumulator.archive_text == "REASONING:\nthink\n\nANSWER:\nhé"


def test_sse_parser_accepts_multiline_data_events():
    accumulator = SSEAccumulator()
    accumulator.feed(b'data: {"choices": [{"delta":\n')
    accumulator.feed(b'data: {"content": "hello"}}]}\n\n')

    assert accumulator.visible_content == "hello"


def test_response_headers_preserve_relevant_metadata_but_remove_hop_by_hop():
    headers = response_headers(
        {
            "content-type": "text/event-stream",
            "cache-control": "no-cache",
            "x-request-id": "request-1",
            "retry-after": "10",
            "content-length": "123",
            "connection": "keep-alive",
            "transfer-encoding": "chunked",
        }
    )

    assert headers == {
        "content-type": "text/event-stream",
        "cache-control": "no-cache",
        "x-request-id": "request-1",
        "retry-after": "10",
    }
