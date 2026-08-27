# Contributor Guide

## Scope

This repository contains an OpenAI-compatible RAG gateway in front of 9Router. The gateway reduces forwarded conversation history, archives older messages in Milvus/Zilliz, retrieves session-scoped memory, and proxies every non-chat route unchanged in purpose.

## Rules

- Use English for source, comments, tests, and documentation.
- Never add production credentials or log secret values.
- Read configuration from environment variables; update `.env.example` for new variables.
- Use strict red-green-refactor TDD for behavior changes. Run the focused failing test before implementation, then the full suite.
- Keep context reduction deterministic. Preserve system/developer messages and valid assistant/tool-call groups.
- Preserve upstream HTTP error statuses and close HTTP clients and streaming responses in every path.
- Schema reset must remain explicit opt-in. Never make startup destructive.
- Do not claim hallucination prevention or token/cost savings without measurements.
- Do not commit or push unless explicitly asked.

## Verification

```bash
python -m pytest -q
python -m compileall -q rag_gateway src plugins tests
```

Use Python 3.10 or 3.11, matching CI.
