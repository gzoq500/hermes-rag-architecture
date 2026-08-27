# Hermes RAG Architecture

[![CI](https://github.com/KezemGolem/hermes-rag-architecture/actions/workflows/ci.yml/badge.svg)](https://github.com/KezemGolem/hermes-rag-architecture/actions/workflows/ci.yml)

An OpenAI-compatible FastAPI gateway that sits between Hermes Agent and 9Router. It forwards a bounded, structurally valid recent context, archives older user/assistant/tool messages in Zilliz Cloud (Milvus), and recalls session-scoped memories using Gemini embeddings.

This project can reduce the amount of conversation history sent upstream when the input exceeds the configured suffix. Actual token, latency, quality, and cost changes depend on workload and configuration and must be measured. Retrieval does not guarantee factual accuracy or eliminate hallucinations.

## Architecture

- **RAG gateway:** `0.0.0.0:20128` by default.
- **9Router:** shifted to `0.0.0.0:20130`.
- **Chat endpoint:** `/v1/chat/completions` is reduced, recalled, archived, and proxied.
- **Other routes:** transparently proxied to 9Router; FastAPI docs/OpenAPI routes are disabled so `/docs`, `/openapi.json`, and all other non-chat paths reach 9Router.
- **Memory:** deterministic record IDs plus Milvus upsert avoid duplicate records when clients resend full history.
- **Schema:** creation is idempotent and non-destructive unless reset is explicitly enabled.

### Context policy

The gateway always preserves `system` and `developer` messages. It also keeps a configurable recent conversation suffix (`RAG_RECENT_MESSAGES`) and expands the boundary when necessary to keep assistant tool calls with their tool results. Older user, assistant, and tool messages are queued for bounded background ingestion rather than forwarded upstream. Retrieved memory is inserted as a developer message with real newline characters.

## Requirements

- Python 3.10 or 3.11
- 9Router
- A Zilliz Cloud or Milvus endpoint
- A Gemini API key with access to the configured embedding model

No production credentials are included. Do not place secrets in source files or service units.

## Install

```bash
git clone https://github.com/KezemGolem/hermes-rag-architecture.git
cd hermes-rag-architecture
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
sudo install -m 0600 .env.example /etc/hermes-rag.env
```

Edit `/etc/hermes-rag.env` and replace only the credential/endpoint examples. The process fails at startup with the names of missing required variables; it never includes their values in the error.

Required variables:

- `ZILLIZ_URI`
- `ZILLIZ_TOKEN`
- `GEMINI_API_KEY`

All supported tunables and defaults are documented in `.env.example`, including retrieval count, recent suffix size, chunk size, ingestion bound, timeouts, ports, model, dimensions, database path, and explicit schema reset.

## Create or verify the collection

Run from the repository root:

```bash
set -a
. /etc/hermes-rag.env
set +a
.venv/bin/python -m src.create_schema_gemini
```

If the collection already exists, this command leaves it unchanged. To intentionally delete and recreate it, set `RAG_RESET_COLLECTION=true` for that invocation only. Reset destroys the collection's existing records.

## systemd

Install the service examples after placing the repository at `/opt/hermes-rag-architecture`:

```bash
sudo cp src/9router-shifted.service /etc/systemd/system/9router.service
sudo cp src/rag-worker.service /etc/systemd/system/rag-worker.service
sudo systemctl daemon-reload
sudo systemctl enable --now 9router.service rag-worker.service
```

`rag-worker.service` reads `/etc/hermes-rag.env` through `EnvironmentFile=`. `9router-shifted.service` listens on port `20130`; the gateway listens on `20128`.

## Session isolation

Session resolution uses this priority:

1. `X-Hermes-Session-Id` request header
2. OpenAI-compatible payload `user`
3. Parameterized lookup of the latest matching `role = 'user'` row in Hermes SQLite
4. `default_session` fallback, with a warning

Exact isolation depends on clients sending `X-Hermes-Session-Id` or `user`. The SQLite fallback is robust for ordinary transcript requests but content matching can be ambiguous when identical user messages exist in multiple sessions.

### Optional Hermes 0.20.5 middleware plugin

The included `plugins/hermes-rag-session-header` plugin uses the verified Hermes 0.20.5 `llm_request` middleware contract. It copies middleware `session_id` into `extra_headers.X-Hermes-Session-Id` without patching Hermes core.

```bash
mkdir -p "${HERMES_HOME:-$HOME/.hermes}/plugins/hermes-rag-session-header"
cp plugins/hermes-rag-session-header/{__init__.py,plugin.yaml} \
  "${HERMES_HOME:-$HOME/.hermes}/plugins/hermes-rag-session-header/"
hermes plugins enable hermes-rag-session-header
```

Restart the Hermes process/gateway after enabling it. Confirm enablement with `hermes plugins`. This plugin contract was checked against installed Hermes Agent `0.20.5`; re-check the official Hermes plugin/middleware documentation when using another version.

## Streaming behavior

For streaming chat requests, the gateway opens the upstream response before returning SSE. Upstream 401 and 5xx responses retain their status and body rather than becoming HTTP 200 streams. The incremental parser tolerates UTF-8 and SSE records split across arbitrary network chunks, captures visible assistant content and common reasoning fields for archival, and closes upstream response/client resources on completion, disconnect, or error.

## Development and tests

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q rag_gateway src plugins tests
```

CI runs the same pytest and compileall checks on Python 3.10 and 3.11. See `AGENTS.md` and `CLAUDE.md` for contributor rules.

## Operational notes

- Old-context ingestion runs after the response path is established and is bounded by `RAG_MAX_INGEST_RECORDS_PER_REQUEST`; failures are logged without failing inference.
- Existing deterministic IDs are queried before embedding so repeated request history is not embedded/upserted again.
- Retrieval itself remains part of request latency because relevant memories must be available before the upstream call.
- Logs report outcomes and exception classes, not credential values or request content.
- Measure upstream input size and end-to-end quality on your own workload before choosing context/retrieval settings.
