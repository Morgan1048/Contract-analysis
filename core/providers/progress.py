"""Transport progress independent of Streamlit and inherited by graph contexts."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class WaitUpdate:
    provider: str
    model: str
    reason: str
    remaining_s: float
    retry: int
    max_retries: int
    http_status: int | None = None
    phase: str = "waiting"


_observer: ContextVar[Callable[[WaitUpdate], None] | None] = ContextVar("backend_wait_observer", default=None)


@contextmanager
def backend_wait_updates(callback: Callable[[WaitUpdate], None]):
    token = _observer.set(callback)
    try:
        yield
    finally:
        _observer.reset(token)


def observer():
    return _observer.get()
