import importlib.util
from pathlib import Path


PLUGIN = Path(__file__).parents[1] / "plugins" / "hermes-rag-session-header" / "__init__.py"


def load_plugin():
    spec = importlib.util.spec_from_file_location("hermes_rag_session_header", PLUGIN)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_middleware_injects_session_header_without_losing_existing_headers():
    plugin = load_plugin()

    result = plugin.add_session_header(
        session_id="session-123",
        request={"messages": [], "extra_headers": {"X-Existing": "value"}},
    )

    assert result["request"]["extra_headers"] == {
        "X-Existing": "value",
        "X-Hermes-Session-Id": "session-123",
    }


def test_middleware_is_noop_without_session_context():
    plugin = load_plugin()

    assert plugin.add_session_header(session_id="", request={"messages": []}) is None


class FakeContext:
    def __init__(self):
        self.registration = None

    def register_middleware(self, kind, callback):
        self.registration = (kind, callback)


def test_plugin_registers_verified_llm_request_middleware_contract():
    plugin = load_plugin()
    context = FakeContext()

    plugin.register(context)

    assert context.registration == ("llm_request", plugin.add_session_header)
