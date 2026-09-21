"""One OTel tree, exported locally and optionally to a local Langfuse dashboard."""
from __future__ import annotations

import atexit
import base64
import json
import logging
import os
from pathlib import Path
import re
import sys
import threading
from typing import Any
from urllib.parse import urlsplit

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor, SpanExporter, SpanExportResult,
)

from .langfuse import config as langfuse_config
from .logging import PrivateRotatingFileHandler, get_logger

_enabled = False
_tracer = None
_provider = None
_configured = False
_lock = threading.RLock()
_ATTRIBUTE_NAMES = {
    "model": "gen_ai.request.model",
    "prompt_tokens": "gen_ai.usage.input_tokens",
    "completion_tokens": "gen_ai.usage.output_tokens",
    "system": "gen_ai.system",
    "observation_type": "langfuse.observation.type",
    "input": "langfuse.observation.input",
    "output": "langfuse.observation.output",
}


def attribute_name(key: str) -> str:
    return _ATTRIBUTE_NAMES.get(key, key)


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"((?:sk-|gsk_)[A-Za-z0-9_-]+|(?:Bearer|Basic)\s+[A-Za-z0-9._=+/-]+|api[_-]?key[=:]\s*\S+)",
                      "[REDACTED]", value, flags=re.I)
    if isinstance(value, dict):
        return {key: "[REDACTED]" if re.search(r"(^|_)(api_key|secret_key|password|authorization)$", key, re.I)
                else _redact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


class _FileHandler(PrivateRotatingFileHandler):
    def handleError(self, record):
        raise


class JsonFileSpanExporter(SpanExporter):
    def __init__(self, path="logs/otel-spans.jsonl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.handler = _FileHandler(path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
        self.handler.setFormatter(logging.Formatter("%(message)s"))

    def export(self, spans):
        try:
            for item in spans:
                self.handler.handle(logging.LogRecord("otel", logging.INFO, "", 0, item.to_json(indent=None), (), None))
            return SpanExportResult.SUCCESS
        except (OSError, ValueError):
            get_logger().error("obs:file_export_failed", extra={"fields": {"path": self.handler.baseFilename}})
            return SpanExportResult.FAILURE

    def shutdown(self):
        self.handler.close()


def configure_otel(service_name="contract-analyzer", enabled: bool | None = None):
    global _configured, _enabled, _provider, _tracer
    with _lock:
        if _provider is not None:
            _provider.shutdown()
        _configured, _provider, _tracer = True, None, None
        lf = None
        if enabled is not False:
            try:
                lf = langfuse_config()
            except ValueError as exc:
                get_logger().error("obs:langfuse_configuration", extra={"fields": {"error": str(exc)}})
        _enabled = enabled if enabled is not None else (
            lf is not None or os.getenv("CA_OTEL_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
        )
        if not _enabled:
            return
        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter(
            out=sys.stderr, formatter=lambda item: item.to_json(indent=None) + "\n",
        )))
        try:
            provider.add_span_processor(SimpleSpanProcessor(JsonFileSpanExporter()))
        except OSError as exc:
            get_logger().error("obs:file_configuration", extra={"fields": {"error": str(exc)}})
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from requests import Session

        def local_exporter(url, headers=None):
            session = Session()
            session.trust_env = False
            return OTLPSpanExporter(endpoint=url, headers=headers, timeout=3, session=session)

        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if endpoint:
            parsed = urlsplit(endpoint)
            if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"} or parsed.username or parsed.password:
                get_logger().error("obs:collector_configuration", extra={"fields": {
                    "error": "Only loopback OTLP collectors are allowed for confidential traces.",
                }})
            elif lf is None or endpoint.rstrip("/") != f"{lf.base_url}/api/public/otel/v1/traces":
                provider.add_span_processor(BatchSpanProcessor(local_exporter(endpoint)))
        if lf is not None:
            auth = base64.b64encode(f"{lf.public_key}:{lf.secret_key}".encode()).decode()
            provider.add_span_processor(BatchSpanProcessor(local_exporter(
                f"{lf.base_url}/api/public/otel/v1/traces",
                headers={"Authorization": f"Basic {auth}", "x-langfuse-ingestion-version": "4"},
            ), schedule_delay_millis=1000))
        # A dedicated provider avoids replacing or duplicating another library's global provider.
        _provider = provider
        _tracer = provider.get_tracer("contract-analyzer")


def flush(timeout_millis=5000):
    return _provider.force_flush(timeout_millis) if _provider is not None else True


def shutdown():
    global _provider
    if _provider is not None:
        _provider.shutdown()
        _provider = None


atexit.register(shutdown)


def ensure_configured():
    with _lock:
        if not _configured:
            configure_otel()


def observation_type(name, fields):
    if fields.get("observation_type"):
        return fields["observation_type"]
    if name.startswith("agent:"):
        return "evaluator" if name == "agent:critic" else "guardrail" if name == "agent:verify" else "agent"
    if name.startswith(("tool:", "chat:tool:")):
        return "tool"
    if name == "backend:call":
        return "span"
    return "generation" if fields.get("model") else "span"


def set_attributes(target, fields):
    generation = target.attributes.get("langfuse.observation.type") in {"generation", "embedding"}
    for key, value in fields.items():
        if value is None:
            continue
        value = _redact({key: value})[key]
        name = attribute_name(key)
        if key in {"prompt_tokens", "completion_tokens", "cost_usd"} and not generation:
            name = f"langfuse.observation.metadata.{key}"
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, ensure_ascii=True)
        target.set_attribute(name, value)
        if key in {"run_id", "question_id", "sub_requirement_id", "round", "stop_reason"}:
            target.set_attribute(f"langfuse.observation.metadata.{key}", value)
        if key == "run_id":
            target.set_attribute("langfuse.session.id", value)
            target.set_attribute("langfuse.trace.metadata.run_id", value)
        if key == "cost_usd" and generation:
            target.set_attribute("langfuse.observation.cost_details", json.dumps({"total": value}))
    if fields.get("failed"):
        from opentelemetry.trace import Status, StatusCode
        target.set_status(Status(StatusCode.ERROR, str(fields.get("stop_reason") or "Operation failed")))


def start_span(name, fields):
    if not _enabled or _tracer is None:
        return None
    result = _tracer.start_span(name)
    result.set_attribute("langfuse.observation.type", observation_type(name, fields))
    result.set_attribute("langfuse.environment", os.getenv("CA_OBS_ENVIRONMENT", "development"))
    set_attributes(result, fields)
    return result
