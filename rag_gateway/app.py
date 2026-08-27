"""FastAPI application and transparent 9Router proxy."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import Settings
from .context import build_recalled_message, extract_text, reduce_context
from .memory import archive_messages, search_memory
from .sessions import resolve_session_id
from .streaming import SSEAccumulator, response_headers

logger = logging.getLogger(__name__)

HOP_BY_HOP_REQUEST_HEADERS = {
    "host",
    "content-length",
    "connection",
    "transfer-encoding",
    "accept-encoding",
}


def _request_headers(request: Request) -> dict[str, str]:
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP_REQUEST_HEADERS
    }


def _assistant_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return ""
    visible = extract_text(message.get("content"))
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if isinstance(reasoning, str) and reasoning and visible:
        return f"REASONING:\n{reasoning}\n\nANSWER:\n{visible}"
    return visible or (reasoning if isinstance(reasoning, str) else "")


def create_app(
    settings: Settings,
    milvus_client: Any,
    embedder: Any,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def archive(session_id: str, messages: list[dict[str, Any]]) -> None:
        if not messages:
            return
        archive_messages(
            milvus_client,
            embedder,
            session_id,
            messages,
            int(time.time()),
            settings.minimum_ingest_chars,
            collection_name=settings.collection_name,
            chunk_max_chars=settings.chunk_max_chars,
            max_records=settings.max_ingest_records_per_request,
        )

    def safe_archive(session_id: str, messages: list[dict[str, Any]]) -> None:
        try:
            archive(session_id, messages)
        except Exception as exc:
            logger.warning("Background memory archival failed: %s", exc.__class__.__name__)

    async def proxy_chat(request: Request, background_tasks: BackgroundTasks) -> Response:
        try:
            payload = json.loads((await request.body()).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return JSONResponse({"error": "Invalid JSON request body"}, status_code=400)
        if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
            return JSONResponse({"error": "messages must be an array"}, status_code=400)
        messages = [message for message in payload["messages"] if isinstance(message, dict)]
        latest_user = next(
            (
                extract_text(message.get("content"))
                for message in reversed(messages)
                if message.get("role") == "user"
            ),
            "",
        )
        session_id = resolve_session_id(request.headers, payload, latest_user, settings.db_path)
        forwarded, archived = reduce_context(messages, settings.recent_messages)
        if archived:
            background_tasks.add_task(safe_archive, session_id, archived)
        if len(latest_user) >= settings.retrieval_min_chars:
            recalled = search_memory(
                milvus_client,
                embedder,
                settings.collection_name,
                latest_user,
                session_id,
                settings.retrieval_limit,
            )
            if recalled:
                insert_at = next(
                    (index for index, message in enumerate(forwarded) if message.get("role") not in {"system", "developer"}),
                    len(forwarded),
                )
                forwarded.insert(insert_at, build_recalled_message(recalled))
        payload["messages"] = forwarded
        client = httpx.AsyncClient(
            base_url=settings.router_base_url,
            transport=transport,
            timeout=settings.upstream_timeout_seconds,
        )
        upstream_request = client.build_request(
            "POST",
            "/v1/chat/completions",
            json=payload,
            headers=_request_headers(request),
        )
        try:
            upstream = await client.send(upstream_request, stream=bool(payload.get("stream")))
        except httpx.HTTPError as exc:
            await client.aclose()
            logger.warning("9Router request failed: %s", exc.__class__.__name__)
            return JSONResponse({"error": "9Router is unavailable"}, status_code=502)
        headers = response_headers(upstream.headers)
        if upstream.status_code >= 400 or not payload.get("stream"):
            try:
                body = await upstream.aread()
                if upstream.status_code < 400:
                    try:
                        result = json.loads(body)
                    except json.JSONDecodeError:
                        result = {}
                    assistant = _assistant_text(result) if isinstance(result, dict) else ""
                    if assistant:
                        background_tasks.add_task(
                            safe_archive,
                            session_id,
                            [{"role": "assistant", "content": assistant}],
                        )
                return Response(
                    body,
                    status_code=upstream.status_code,
                    headers=headers,
                    background=background_tasks,
                )
            finally:
                await upstream.aclose()
                await client.aclose()

        accumulator = SSEAccumulator()

        async def forward_stream():
            try:
                async for chunk in upstream.aiter_bytes():
                    accumulator.feed(chunk)
                    yield chunk
                accumulator.finish()
                if accumulator.archive_text:
                    background_tasks.add_task(
                        safe_archive,
                        session_id,
                        [{"role": "assistant", "content": accumulator.archive_text}],
                    )
            except BaseException:
                logger.info("Streaming client disconnected or upstream stream failed")
                raise
            finally:
                await upstream.aclose()
                await client.aclose()

        return StreamingResponse(
            forward_stream(),
            status_code=upstream.status_code,
            headers=headers,
            media_type=None,
            background=background_tasks,
        )

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH", "TRACE"],
    )
    async def catch_all(
        request: Request, path: str, background_tasks: BackgroundTasks
    ) -> Response:
        if path == "v1/chat/completions" and request.method == "POST":
            return await proxy_chat(request, background_tasks)
        url = f"/{path}"
        if request.url.query:
            url += f"?{request.url.query}"
        try:
            async with httpx.AsyncClient(
                base_url=settings.router_base_url,
                transport=transport,
                timeout=settings.upstream_timeout_seconds,
            ) as client:
                upstream = await client.request(
                    request.method,
                    url,
                    content=await request.body(),
                    headers=_request_headers(request),
                    follow_redirects=False,
                )
        except httpx.HTTPError as exc:
            logger.warning("9Router route proxy failed: %s", exc.__class__.__name__)
            return JSONResponse({"error": "9Router is unavailable"}, status_code=502)
        return Response(
            upstream.content,
            status_code=upstream.status_code,
            headers=response_headers(upstream.headers),
        )

    return app
