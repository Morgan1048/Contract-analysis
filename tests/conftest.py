"""Offline tests must not export fixtures to any configured observability server."""
import pytest


@pytest.fixture(autouse=True)
def isolate_observability(monkeypatch):
    monkeypatch.setenv("CA_LANGFUSE_ENABLED", "0")
    monkeypatch.setenv("CA_OTEL_ENABLED", "0")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
