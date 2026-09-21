from __future__ import annotations


class BackendError(Exception):
    """An LLM/embedding backend could not be reached or answered incorrectly at
    the transport/interface level: network failures, timeouts, non-2xx HTTP
    responses (including exhausted retries), and client interface mismatches
    (e.g. an unexpected keyword argument).

    This is deliberately **not** a subclass of ``ValueError``/``TypeError``: a
    ``BackendError`` must never be caught by code that treats those exceptions
    as "the model produced malformed output" (a legitimate evidence-level
    outcome). A backend failure means "we do not know the answer"; it must
    never be silently downgraded into a compliance verdict such as NOT_FOUND.
    """


class RequestBudgetExceeded(BackendError):
    """Transport cannot complete or honor a retry inside the caller's deadline."""
