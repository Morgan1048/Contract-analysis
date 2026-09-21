"""Acceptance tests for core/obs/otel.py (task 10, Part B).

Test 15 is non-negotiable: mcp_server.py serves JSON-RPC on stdout, and a span
printed there would corrupt the protocol.
"""
from __future__ import annotations

import io
import json
import logging

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from core.obs import otel
from core.obs.logging import get_logger
from core.obs.trace import span


class _Capture(logging.Handler):
    """Collects (event name, fields dict) for every log record emitted during a test."""

    def __init__(self):
        super().__init__()
        self.entries: list[tuple[str, dict]] = []

    def emit(self, record):
        self.entries.append((record.msg, dict(getattr(record, "fields", {}))))


def _enable_memory_exporter() -> InMemorySpanExporter:
    """Enable OTel and attach an in-memory exporter to the active tracer provider."""
    otel.configure_otel(service_name="test-otel", enabled=True)
    exporter = InMemorySpanExporter()
    otel._provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


@pytest.fixture(autouse=True)
def _reset_otel_state():
    yield
    otel.configure_otel(enabled=False)


@pytest.fixture(scope="module", autouse=True)
def _shutdown_tracer_provider_once():
    # Flush and stop the BatchSpanProcessor's background thread before pytest
    # tears down captured streams, otherwise it logs a harmless but noisy
    # "I/O operation on closed file" traceback at interpreter shutdown.
    yield
    provider = otel._provider
    if hasattr(provider, "shutdown"):
        provider.shutdown()


def test_disabled_otel_leaves_spans_and_logs_byte_identical_to_before():
    otel.configure_otel(enabled=False)
    logger = get_logger()
    capture = _Capture()
    logger.addHandler(capture)
    try:
        with span("otel-regression-check"):
            pass
    finally:
        logger.removeHandler(capture)

    assert [name for name, _ in capture.entries] == ["otel-regression-check", "otel-regression-check"]
    start_fields, end_fields = capture.entries[0][1], capture.entries[1][1]
    assert start_fields["message"] == "start"
    assert set(start_fields) == {"trace_id", "span_id", "parent_span_id", "message"}
    assert end_fields["message"] == "end"
    assert set(end_fields) == {"trace_id", "span_id", "parent_span_id", "message", "duration_ms"}


def test_enabled_otel_span_names_and_nesting_match_the_json_log():
    exporter = _enable_memory_exporter()
    logger = get_logger()
    capture = _Capture()
    logger.addHandler(capture)
    try:
        with span("outer"):
            with span("inner"):
                pass
    finally:
        logger.removeHandler(capture)

    logged_starts = [name for name, fields in capture.entries if fields.get("message") == "start"]
    assert logged_starts == ["outer", "inner"]

    finished = {s.name: s for s in exporter.get_finished_spans()}
    assert set(finished) == {"outer", "inner"}
    assert finished["inner"].parent.span_id == finished["outer"].context.span_id


def test_nothing_is_ever_written_to_stdout(capsys):
    for enabled in (False, True):
        otel.configure_otel(enabled=enabled)
        with span("stdout-check", model="gpt-4o-mini", prompt_tokens=10, completion_tokens=5):
            pass
        assert capsys.readouterr().out == ""


def test_mcp_stdio_round_trip_writes_only_json_rpc_to_stdout(tmp_path, capsys):
    from core.store import RunStore
    from core.tools.contract_tools import build_mcp_registry
    from mcp_server import dispatch

    otel.configure_otel(enabled=True)
    registry = build_mcp_registry(store=RunStore(tmp_path / "runs.db"))
    stdout = io.StringIO()
    response = dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}, registry)
    stdout.write(json.dumps(response) + "\n")

    assert capsys.readouterr().out == ""
    parsed = json.loads(stdout.getvalue())
    assert parsed["jsonrpc"] == "2.0"
    assert "result" in parsed


def test_gen_ai_attributes_carry_real_token_counts_not_estimates():
    exporter = _enable_memory_exporter()
    with span("llm-call", model="gpt-4o-mini", prompt_tokens=123, completion_tokens=45, system="openai"):
        pass

    llm_span = next(s for s in exporter.get_finished_spans() if s.name == "llm-call")
    assert llm_span.attributes["gen_ai.request.model"] == "gpt-4o-mini"
    assert llm_span.attributes["gen_ai.usage.input_tokens"] == 123
    assert llm_span.attributes["gen_ai.usage.output_tokens"] == 45
    assert llm_span.attributes["gen_ai.system"] == "openai"


def test_span_that_raises_still_records_error_and_reraises_through_both_paths():
    exporter = _enable_memory_exporter()
    logger = get_logger()
    capture = _Capture()
    logger.addHandler(capture)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            with span("failing-op"):
                raise RuntimeError("boom")
    finally:
        logger.removeHandler(capture)

    error_records = [fields for name, fields in capture.entries if fields.get("message") == "RuntimeError"]
    assert error_records, "expected an error-level log record for the raised exception"

    failing_span = next(s for s in exporter.get_finished_spans() if s.name == "failing-op")
    assert failing_span.status.status_code == StatusCode.ERROR
    assert any(event.name == "exception" for event in failing_span.events)


def test_secrets_are_redacted_in_otel_attributes_as_they_are_in_log_records():
    exporter = _enable_memory_exporter()
    with span("secret-op", api_key="sk-abcdef1234567890"):
        pass

    secret_span = next(s for s in exporter.get_finished_spans() if s.name == "secret-op")
    assert secret_span.attributes["api_key"] == "[REDACTED]"
