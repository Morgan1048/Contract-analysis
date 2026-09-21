import json
from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from core.obs import langfuse, otel
from core.obs.trace import span, set_span_attributes


@pytest.fixture
def captured(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    otel.configure_otel(enabled=True)
    exporter = InMemorySpanExporter()
    otel._provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield exporter
    otel.configure_otel(enabled=False)


def test_native_ids_metadata_redaction_and_local_span_file(captured):
    with span("analyze", run_id="run-test") as root_id:
        with span("agent:judge", question_id="Q1"):
            set_span_attributes(prompt_tokens=100, cost_usd=.5)
            with span("llm:generate", observation_type="generation", model="test") as generation_id:
                set_span_attributes(input={"authorization": "Basic hidden", "text": "api_key=private"},
                                    output="sk-private", prompt_tokens=100, completion_tokens=12, cost_usd=.5)
    spans = {s.name: s for s in captured.get_finished_spans()}
    root, node, generation = (spans[n] for n in ("analyze", "agent:judge", "llm:generate"))
    assert root_id == f"{root.context.span_id:016x}"
    assert generation_id == f"{generation.context.span_id:016x}"
    assert generation.parent.span_id == node.context.span_id
    assert generation.attributes["langfuse.session.id"] == "run-test"
    assert generation.attributes["langfuse.observation.metadata.question_id"] == "Q1"
    assert generation.attributes["gen_ai.usage.input_tokens"] == 100
    assert json.loads(generation.attributes["langfuse.observation.cost_details"]) == {"total": .5}
    assert "gen_ai.usage.input_tokens" not in node.attributes
    assert "langfuse.observation.cost_details" not in node.attributes
    saved = Path("logs/otel-spans.jsonl").read_text()
    assert Path("logs/otel-spans.jsonl").stat().st_mode & 0o777 == 0o600
    assert len(saved.splitlines()) == 3
    assert "private" not in saved and "Basic hidden" not in saved


@pytest.mark.parametrize("base", [
    "https://cloud.langfuse.com", "http://127.0.0.1.evil.test", "http://user:pass@localhost",
    "http://localhost/path", "http://localhost?key=secret",
])
def test_hosted_or_credentialed_urls_rejected(monkeypatch, base):
    monkeypatch.setenv("CA_LANGFUSE_ENABLED", "1")
    monkeypatch.setenv("LANGFUSE_BASE_URL", base)
    with pytest.raises(ValueError, match="loopback"):
        langfuse.config()


def test_missing_configuration_is_explicit(monkeypatch):
    monkeypatch.setattr(langfuse, "dotenv_values", lambda _: {})
    monkeypatch.setenv("CA_LANGFUSE_ENABLED", "1")
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_PROJECT_ID"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="requires"):
        langfuse.config()


def test_generation_usage_recorded_even_for_invalid_json(captured, monkeypatch):
    from core.llm.client import LLMClient, LLMClientError
    from core.providers.config import ProviderConfig
    client = LLMClient(ProviderConfig("ollama", "http://localhost:11434/v1", "test"))
    monkeypatch.setattr(client._client, "complete_with_usage",
                        lambda *_args, **_kwargs: ("invalid", {"prompt_tokens": 15, "completion_tokens": 3}))
    with pytest.raises(LLMClientError):
        client.complete_json("Test system", "Independent fixture", {})
    generation = next(s for s in captured.get_finished_spans() if s.name == "llm:generate")
    assert generation.attributes["gen_ai.usage.input_tokens"] == 15
    assert generation.status.status_code.name == "ERROR"


def test_oversized_prompt_never_reaches_provider(monkeypatch):
    from core.llm.client import LLMClient, PromptBudgetExceeded
    from core.providers.config import ProviderConfig
    client = LLMClient(ProviderConfig("ollama", "http://localhost:11434/v1", "test", max_input_tokens=250))
    calls = []
    monkeypatch.setattr(client._client, "complete_with_usage", lambda *_a, **_k: calls.append(True))
    with pytest.raises(PromptBudgetExceeded, match="prompt_token_budget"):
        client.complete_json("Read the evidence", "independent fixture " * 1000, {})
    assert calls == []


def test_initialize_private_credentials_is_idempotent(tmp_path):
    from scripts.start_langfuse import initialize
    path = tmp_path / ".env.langfuse"
    initialize(path)
    original = path.read_bytes()
    initialize(path)
    assert path.read_bytes() == original
    assert path.stat().st_mode & 0o777 == 0o600


def test_dashboard_links_do_not_probe_backend_or_reveal_secrets(monkeypatch):
    from ui.monitoring_view import _render_langfuse
    from types import SimpleNamespace
    from ui import monitoring_view
    cfg = langfuse.LangfuseConfig("http://127.0.0.1:3100", "project", "public", "private")
    monkeypatch.setattr(monitoring_view, "langfuse_config", lambda: cfg)
    messages = []
    st = SimpleNamespace(subheader=messages.append, markdown=messages.append, caption=messages.append)
    store = SimpleNamespace(list_runs=lambda limit: [{"run_id": "run-test"}],
                            get_run=lambda _: SimpleNamespace(usage={"trace_id": "a" * 32}))
    _render_langfuse(store, st)
    assert cfg.trace_url("a" * 32) in str(messages)
    assert "private" not in str(messages)
