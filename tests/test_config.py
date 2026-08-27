import pytest

from rag_gateway.config import ConfigurationError, Settings


def valid_environment():
    return {
        "ZILLIZ_URI": "https://example.zillizcloud.com:19532",
        "ZILLIZ_TOKEN": "top-secret-zilliz",
        "GEMINI_API_KEYS": "top-secret-gemini,key2",
    }


def test_settings_fail_clearly_when_required_credentials_are_missing():
    with pytest.raises(ConfigurationError, match="ZILLIZ_URI, ZILLIZ_TOKEN, GEMINI_API_KEYS"):
        Settings.from_env({})


def test_settings_read_tunables_and_do_not_expose_secrets():
    environment = valid_environment() | {
        "RAG_RECENT_MESSAGES": "12",
        "RAG_RETRIEVAL_LIMIT": "7",
        "RAG_MAX_INGEST_RECORDS_PER_REQUEST": "8",
        "RAG_RESET_COLLECTION": "true",
        "ROUTER_BASE_URL": "http://127.0.0.1:20130",
    }

    settings = Settings.from_env(environment)

    assert settings.recent_messages == 12
    assert settings.retrieval_limit == 7
    assert settings.max_ingest_records_per_request == 8
    assert settings.reset_collection is True
    assert settings.router_base_url == "http://127.0.0.1:20130"
    rendered = repr(settings)
    assert "top-secret-zilliz" not in rendered
    assert "top-secret-gemini" not in rendered
    assert "key2" not in rendered


def test_invalid_numeric_tunable_has_actionable_error():
    environment = valid_environment() | {"RAG_RECENT_MESSAGES": "zero"}

    with pytest.raises(ConfigurationError, match="RAG_RECENT_MESSAGES must be an integer"):
        Settings.from_env(environment)
