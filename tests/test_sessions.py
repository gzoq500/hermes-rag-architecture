import logging
import sqlite3

from rag_gateway.sessions import lookup_session_id, resolve_session_id


def make_database(path):
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT)"
    )
    connection.executemany(
        "INSERT INTO messages(session_id, role, content) VALUES (?, ?, ?)",
        [
            ("assistant-session", "assistant", "Find 100%_coverage please"),
            ("wrong-user", "user", "Find 100XXcoverage please"),
            ("right-user", "user", "prefix Find 100%_coverage please suffix"),
        ],
    )
    connection.commit()
    connection.close()


def test_sqlite_lookup_matches_user_content_and_escapes_like_wildcards(tmp_path, caplog):
    database = tmp_path / "state.db"
    make_database(database)

    with caplog.at_level(logging.INFO):
        result = lookup_session_id(str(database), "Find 100%_coverage please")

    assert result == "right-user"
    assert "Resolved session from SQLite" in caplog.text
    assert "Find 100%_coverage" not in caplog.text


def test_sqlite_lookup_logs_no_match_explicitly(tmp_path, caplog):
    database = tmp_path / "state.db"
    make_database(database)

    with caplog.at_level(logging.INFO):
        result = lookup_session_id(str(database), "not present")

    assert result is None
    assert "No SQLite session match" in caplog.text


def test_session_resolution_prefers_header_then_payload_then_sqlite(tmp_path):
    database = tmp_path / "state.db"
    make_database(database)

    assert resolve_session_id(
        {"X-Hermes-Session-Id": "header-session"},
        {"user": "payload-session"},
        "Find 100%_coverage please",
        str(database),
    ) == "header-session"
    assert resolve_session_id(
        {}, {"user": "payload-session"}, "Find 100%_coverage please", str(database)
    ) == "payload-session"
    assert resolve_session_id(
        {}, {}, "Find 100%_coverage please", str(database)
    ) == "right-user"


def test_session_resolution_uses_default_only_after_all_sources_fail(tmp_path):
    database = tmp_path / "state.db"
    make_database(database)

    assert resolve_session_id({}, {}, "missing", str(database)) == "default_session"
