from rag_gateway.memory import (
    MemoryRecord,
    archive_messages,
    deterministic_record_id,
    escape_milvus_string,
)


class FakeEmbedder:
    def embed(self, text):
        return [float(len(text))]


class FakeMilvus:
    def __init__(self):
        self.upsert_calls = []
        self.existing_ids = set()

    def query(self, **kwargs):
        return [{"id": record_id} for record_id in sorted(self.existing_ids)]

    def upsert(self, **kwargs):
        self.upsert_calls.append(kwargs)
        self.existing_ids.update(record["id"] for record in kwargs["data"])


class CountingEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        return [1.0]


def test_archive_skips_existing_ids_before_embedding_and_bounds_new_work():
    client = FakeMilvus()
    embedder = CountingEmbedder()
    messages = [
        {"role": "user", "content": f"long archived message number {index}"}
        for index in range(5)
    ]

    assert archive_messages(client, embedder, "session", messages, 100, 1, max_records=2) == 2
    assert len(embedder.calls) == 2
    assert archive_messages(client, embedder, "session", messages, 101, 1, max_records=2) == 2
    assert len(embedder.calls) == 4
    assert archive_messages(client, embedder, "session", messages, 102, 1, max_records=10) == 1
    assert len(embedder.calls) == 5
    assert archive_messages(client, embedder, "session", messages, 103, 1, max_records=10) == 0
    assert len(embedder.calls) == 5


def test_archive_logs_query_failure_and_continues(caplog):
    class QueryFailureClient(FakeMilvus):
        def query(self, **kwargs):
            raise RuntimeError("query failed")

    client = QueryFailureClient()
    embedder = CountingEmbedder()

    with caplog.at_level("WARNING"):
        count = archive_messages(
            client,
            embedder,
            "session",
            [{"role": "user", "content": "long enough to archive"}],
            100,
            1,
        )

    assert count == 1
    assert "Existing memory ID lookup failed" in caplog.text
    assert "query failed" not in caplog.text


def test_deterministic_ids_are_stable_and_content_sensitive():
    first = deterministic_record_id("session", "user", "hello", 0)
    second = deterministic_record_id("session", "user", "hello", 0)
    changed = deterministic_record_id("session", "assistant", "hello", 0)

    assert first == second
    assert first != changed
    assert len(first) == 64


def test_archive_uses_upsert_and_deduplicates_repeated_request_history():
    client = FakeMilvus()
    messages = [
        {"role": "user", "content": "a sufficiently long user message"},
        {"role": "assistant", "content": "a sufficiently long assistant message"},
        {"role": "tool", "tool_call_id": "call-1", "content": "a sufficiently long tool result"},
        {"role": "system", "content": "not archived"},
    ]

    archive_messages(client, FakeEmbedder(), "session-a", messages, 100, 5)
    archive_messages(client, FakeEmbedder(), "session-a", messages, 101, 5)

    assert len(client.upsert_calls) == 1
    first_batch = client.upsert_calls[0]["data"]
    assert len(first_batch) == 3
    assert {record["role"] for record in first_batch} == {"user", "assistant", "tool"}


def test_escape_milvus_string_handles_quotes_backslashes_and_controls():
    escaped = escape_milvus_string('session"\\\n\r\t')

    assert escaped == 'session\\"\\\\\\n\\r\\t'


def test_memory_record_handles_structured_content():
    message = {
        "role": "assistant",
        "content": [{"type": "output_text", "text": "structured answer"}],
    }

    records = MemoryRecord.from_message("session", message, 42, chunk_max_chars=8000)

    assert len(records) == 1
    assert records[0].text == "ASSISTANT: structured answer"
