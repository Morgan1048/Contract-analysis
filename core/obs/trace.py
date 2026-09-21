from __future__ import annotations

import contextvars
import time
import uuid
from contextlib import contextmanager

from .logging import get_logger
from .otel import _redact, ensure_configured, set_attributes, start_span

_trace = contextvars.ContextVar("trace_id", default=None)
_parent = contextvars.ContextVar("span_id", default=None)
_span_fields = contextvars.ContextVar("span_fields", default=None)
_otel_span = contextvars.ContextVar("otel_span", default=None)


def current_trace_id():
    value = _trace.get()
    if value is None:
        value = str(uuid.uuid4())
        _trace.set(value)
    return value


def new_run_id():
    return str(uuid.uuid4())


def set_span_attributes(**fields) -> None:
    """Attach attributes learned during a span, such as an LLM response's usage.

    Start records deliberately remain cheap; the corresponding end record and
    OpenTelemetry span carry the completed values, which makes one log record
    sufficient to account for a completed call.
    """
    current = _span_fields.get()
    if current is None:
        raise RuntimeError("set_span_attributes must be called inside a span")
    current.update(fields)
    otel_span = _otel_span.get()
    if otel_span is not None:
        set_attributes(otel_span, fields)


@contextmanager
def span(name: str, **fields):
    ensure_configured()
    span_id = str(uuid.uuid4())
    parent = _parent.get()
    is_root = parent is None
    trace_token = _trace.set(None) if is_root else None
    logger = get_logger()
    inherited = _span_fields.get() or {}
    fields = {**{k: inherited[k] for k in ("run_id", "question_id", "sub_requirement_id", "round")
                 if k in inherited}, **fields}
    otel_span = start_span(name, fields)
    otel_scope = None
    if otel_span is not None:
        span_id = format(otel_span.get_span_context().span_id, "016x")
        from opentelemetry.trace import use_span
        otel_scope = use_span(otel_span, end_on_exit=False)
        otel_scope.__enter__()
        if is_root:
            # Derive our trace_id from OTel's own span context rather than a
            # separate uuid4, so a trace found in Jaeger and its JSON log
            # lines share one native id instead of requiring an attribute
            # lookup to correlate them.
            _trace.set(format(otel_span.get_span_context().trace_id, "032x"))
    token = _parent.set(span_id)
    trace_id = current_trace_id()
    base = dict(fields, trace_id=trace_id, span_id=span_id, parent_span_id=parent)
    fields_token = _span_fields.set(base)
    otel_token = _otel_span.set(otel_span)
    logger.info(name, extra={"fields": dict(base, message="start")})
    started = time.perf_counter()
    try:
        yield span_id
    except Exception as exc:
        base.update(duration_ms=(time.perf_counter() - started) * 1000, message=type(exc).__name__)
        logger.error(name, extra={"fields": base})
        if otel_span is not None:
            from opentelemetry.trace import Status, StatusCode
            otel_span.add_event("exception", {"exception.type": type(exc).__name__,
                                            "exception.message": _redact(str(exc))})
            otel_span.set_status(Status(StatusCode.ERROR, _redact(str(exc))))
            otel_span.end()
        raise
    else:
        base.update(duration_ms=(time.perf_counter() - started) * 1000, message="end")
        logger.info(name, extra={"fields": base})
        if otel_span is not None:
            otel_span.end()
    finally:
        if otel_scope is not None:
            otel_scope.__exit__(None, None, None)
        _otel_span.reset(otel_token)
        _span_fields.reset(fields_token)
        _parent.reset(token)
        if trace_token is not None:
            _trace.reset(trace_token)
