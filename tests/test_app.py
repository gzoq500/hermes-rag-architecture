import json

import httpx
import pytest

from rag_gateway.app import create_app
from rag_gateway.config import Settings


class FakeMilvus:
    def __init__(self):
        self.upserts = []

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def search(self, **kwargs):
        return []

    def query(self, **kwargs):
        return []


class FakeEmbedder:
    def embed(self, text):
        return [1.0]


class OrderingEmbedder(FakeEmbedder):
    def __init__(self, events):
        self.events = events

    def embed(self, text):
        self.events.append("embed")
        return super().embed(text)


def settings(tmp_path):
    return Settings(
        zilliz_uri="https://example.invalid",
        zilliz_token="secret",
        gemini_api_keys=["secret"],
        db_path=str(tmp_path / "missing.db"),
        recent_messages=2,
        retrieval_min_chars=999,
        minimum_ingest_chars=1,
    )


@pytest.mark.asyncio
async def test_streaming_upstream_error_preserves_status_headers_and_body(tmp_path):
    async def upstream(request):
        return httpx.Response(
            401,
            headers={"content-type": "application/json", "x-request-id": "req-401"},
            json={"error": "unauthorized"},
        )

    app = create_app(
        settings(tmp_path),
        FakeMilvus(),
        FakeEmbedder(),
        transport=httpx.MockTransport(upstream),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"stream": True, "messages": [{"role": "user", "content": "hello world"}]},
        )

    assert response.status_code == 401
    assert response.json() == {"error": "unauthorized"}
    assert response.headers["x-request-id"] == "req-401"


@pytest.mark.asyncio
async def test_old_context_archival_does_not_delay_upstream_request(tmp_path):
    events = []

    async def upstream(request):
        events.append("upstream")
        assert "embed" not in events
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    app = create_app(
        settings(tmp_path),
        FakeMilvus(),
        OrderingEmbedder(events),
        transport=httpx.MockTransport(upstream),
    )
    messages = [
        {"role": "user", "content": f"old message {index}"}
        for index in range(20)
    ] + [{"role": "user", "content": "current message"}]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Hermes-Session-Id": "session"},
            json={"messages": messages},
        )

    assert response.status_code == 200
    assert events[0] == "upstream"
    assert "embed" in events


@pytest.mark.asyncio
async def test_chat_reduces_forwarded_context_and_archives_stream_from_split_sse(tmp_path):
    captured = {}
    event = b'data: {"choices":[{"delta":{"content":"answer"}}]}\n\ndata: [DONE]\n\n'

    async def upstream(request):
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream", "cache-control": "no-cache"},
            content=event,
        )

    milvus = FakeMilvus()
    app = create_app(
        settings(tmp_path), milvus, FakeEmbedder(), transport=httpx.MockTransport(upstream)
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "user", "content": "current user"},
    ]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Hermes-Session-Id": "isolated-session"},
            json={"stream": True, "messages": messages},
        )

    assert response.status_code == 200
    assert response.content == event
    assert captured["payload"]["messages"] == [messages[0], messages[2], messages[3]]
    archived_texts = [item["text"] for call in milvus.upserts for item in call["data"]]
    assert any("old user" in text for text in archived_texts)
    assert any("ANSWER:\nanswer" not in text and "answer" in text for text in archived_texts)


@pytest.mark.asyncio
async def test_docs_and_other_non_chat_paths_are_transparently_proxied(tmp_path):
    seen = []

    async def upstream(request):
        seen.append((request.method, request.url.path, request.url.query, request.content))
        return httpx.Response(418, headers={"x-request-id": "route"}, content=b"upstream docs")

    app = create_app(
        settings(tmp_path), FakeMilvus(), FakeEmbedder(), transport=httpx.MockTransport(upstream)
    )
    assert app.docs_url is None
    assert app.openapi_url is None
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/docs?format=raw")

    assert response.status_code == 418
    assert response.content == b"upstream docs"
    assert seen == [("GET", "/docs", b"format=raw", b"")]
