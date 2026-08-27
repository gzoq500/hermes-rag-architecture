# Claude Development Instructions

Follow `AGENTS.md` as the authoritative contributor guide for this repository.

Key constraints:

1. Work test-first and verify every regression with deterministic pytest coverage.
2. Keep all credentials out of source, logs, examples, and test output.
3. Preserve OpenAI message/tool consistency during context reduction.
4. Never turn an upstream authentication or server error into a successful SSE response.
5. Keep Milvus schema management non-destructive unless `RAG_RESET_COLLECTION=true` is explicitly set.
6. Keep all code, comments, and documentation in English.
7. Run pytest and compileall before reporting completion.
