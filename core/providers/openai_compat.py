from __future__ import annotations

import json
import math
import random
import re
import socket
import ssl
import time
import contextvars
import threading
from dataclasses import replace
from contextlib import contextmanager
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import certifi

from core.providers.config import ProviderConfig
from core.providers.errors import BackendError, RequestBudgetExceeded
from core.providers.progress import WaitUpdate, observer
from core.obs.trace import set_span_attributes, span

# X2: retry on rate limiting and transient server/network failures, never on
# a genuine client error (4xx other than 429) which retrying cannot fix.
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_deadline = contextvars.ContextVar("backend_deadline", default=None)


@contextmanager
def request_deadline(seconds: float):
    existing = _deadline.get()
    deadline = time.monotonic() + max(0, seconds)
    token = _deadline.set(min(deadline, existing) if existing is not None else deadline)
    try:
        yield
    finally:
        _deadline.reset(token)


def _remaining():
    deadline = _deadline.get()
    remaining = deadline - time.monotonic() if deadline is not None else float("inf")
    if remaining <= 0:
        raise RequestBudgetExceeded("transport_time_budget")
    return remaining


def _sleep_with_budget(seconds):
    remaining = _remaining()
    if seconds >= remaining:
        raise RequestBudgetExceeded(
            f"retry_after_exceeds_time_budget: required_wait_s={seconds:.1f}, remaining_budget_s={remaining:.1f}"
        )
    time.sleep(seconds)


class OpenAICompatClient:
    """Minimal OpenAI-compatible client for local and hosted backends."""

    def __init__(self, cfg: ProviderConfig) -> None:
        self.cfg = cfg
        # X3: Homebrew/python.org macOS Python does not read the system
        # keychain, so the default SSL context fails CERTIFICATE_VERIFY_FAILED
        # against every real HTTPS backend. Build the context from certifi's
        # bundled CA store instead of relying on the platform trust store.
        self._ssl_context = ssl.create_default_context(cafile=certifi.where())
        self._last_request_at: float = 0.0
        self._cooldown_until = 0.0
        self._cooldown_details: dict = {}
        self._cooldown_lock = threading.Lock()
        if not math.isfinite(cfg.retry_budget_s) or cfg.retry_budget_s <= 0:
            raise ValueError("retry_budget_s must be positive and finite")
        if not math.isfinite(cfg.rate_limit_backoff_s) or cfg.rate_limit_backoff_s <= 0:
            raise ValueError("rate_limit_backoff_s must be positive and finite")

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            # Z2: hosted providers are typically fronted by Cloudflare, which
            # rejects the default urllib UA ("Python-urllib/3.x") with a 403
            # (error 1010) before the request ever reaches the backend.
            "User-Agent": "contract-analyzer/0.1",
        }
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        elif self.cfg.backend in {"ollama", "lmstudio"}:
            headers["Authorization"] = "Bearer local"
        return headers

    def _pace(self) -> None:
        """Enforce a configurable minimum gap between outgoing requests."""
        while True:
            with self._cooldown_lock:
                until, details = self._cooldown_until, dict(self._cooldown_details)
            wait = until - time.monotonic()
            if wait > 0:
                self._wait(wait, **details)
            with self._cooldown_lock:
                if self._cooldown_until <= until:
                    self._cooldown_until = 0.0
                    break
        if self.cfg.min_request_interval_s <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.cfg.min_request_interval_s - elapsed
        if remaining > 0:
            self._wait(remaining, reason="request_pacing")

    def _cooldown(self, seconds: float, **details):
        with self._cooldown_lock:
            until = time.monotonic() + seconds
            if until >= self._cooldown_until:
                self._cooldown_until, self._cooldown_details = until, details

    def _wait(self, seconds: float, *, reason: str, retry: int = 0, http_status: int | None = None):
        event = WaitUpdate(self.cfg.provider, self.cfg.model, reason, seconds,
                           retry, self.cfg.max_retries, http_status)
        notify = observer()
        with span("backend:wait", observation_type="span", provider=self.cfg.provider, model=self.cfg.model,
                  reason=reason, wait_seconds=seconds, retry_count=retry, http_status=http_status):
            available = _remaining()
            if seconds >= available:
                if notify:
                    notify(replace(event, phase="blocked"))
                raise RequestBudgetExceeded(
                    f"retry_after_exceeds_time_budget: HTTP {http_status}, required_wait_s={seconds:.1f}, "
                    f"remaining_budget_s={available:.1f}. Increase the question and retry budgets, or wait for quota reset."
                )
            left = seconds
            while left > 0:
                if notify:
                    notify(replace(event, remaining_s=left))
                step = min(left, 5.0) if notify else left
                _sleep_with_budget(step)
                left -= step
            if notify:
                notify(replace(event, remaining_s=0, phase="resumed"))

    def _retry_delay(self, attempt: int, retry_after: str | None) -> float:
        """Delay before the next attempt, honouring ``Retry-After`` when the
        backend supplied one, else exponential backoff with jitter."""
        parsed = self._retry_after_seconds(retry_after)
        if parsed is not None:
            return parsed
        base = min(self.cfg.backoff_cap_s, self.cfg.backoff_base_s * (2 ** attempt))
        return base * (0.5 + random.random() * 0.5)

    @staticmethod
    def _retry_after_seconds(retry_after):
        if retry_after:
            try:
                delay = float(retry_after)
                if math.isfinite(delay):
                    return max(0.0, delay)
            except ValueError:
                try:
                    dt = parsedate_to_datetime(retry_after)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
                except (TypeError, ValueError):
                    pass
        return None

    @staticmethod
    def _reset_delay(headers):
        waits = []
        for kind in ("requests", "tokens"):
            remaining = headers.get(f"x-ratelimit-remaining-{kind}")
            reset = headers.get(f"x-ratelimit-reset-{kind}", "")
            try:
                exhausted = remaining is not None and float(remaining) <= 0
            except ValueError:
                exhausted = False
            if exhausted and reset and re.fullmatch(r"(?:\d+(?:\.\d+)?(?:ms|s|m|h|d))+", reset):
                scale = {"ms": .001, "s": 1, "m": 60, "h": 3600, "d": 86400}
                waits.append(sum(float(n) * scale[unit] for n, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h|d)", reset)))
        return max(1.0, max(waits)) if waits else None

    def _rate_delay(self, attempt, headers):
        delay = self._retry_after_seconds(headers.get("Retry-After"))
        if delay is not None:
            return max(1.0, delay)
        return self._reset_delay(headers) or min(600.0, self.cfg.rate_limit_backoff_s * 2 ** min(attempt, 10))

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        with span("backend:call", provider=self.cfg.provider, backend=self.cfg.backend,
                  model=self.cfg.model, operation=path):
            with request_deadline(self.cfg.retry_budget_s):
                return self._request_json(path, payload)

    def _request_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        base = self.cfg.base_url.rstrip("/") + "/"
        url = urljoin(base, path.lstrip("/"))
        data = json.dumps(payload).encode("utf-8")
        attempt = 0
        while True:
            _remaining()
            self._pace()
            request = Request(url, data=data, headers=self._headers(), method="POST")
            self._last_request_at = time.monotonic()
            try:
                with urlopen(request, timeout=min(self.cfg.timeout_s, _remaining()), context=self._ssl_context) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    headers = getattr(response, "headers", {}) or {}
                    reset_delay = self._reset_delay(headers)
                    if reset_delay is not None:
                        self._cooldown(reset_delay, reason="quota_reset", retry=0, http_status=200)
                    set_span_attributes(retry_count=attempt, retry_exhausted=False, http_status=200,
                                        rate_limits={k.lower(): v for k, v in headers.items()
                                                     if k.lower().startswith("x-ratelimit-")})
                    return result
            except HTTPError as exc:
                try:
                    body = exc.read().decode("utf-8", errors="replace")
                except Exception:  # pragma: no cover - defensive
                    body = ""
                headers = exc.headers or {}
                try:
                    error = json.loads(body).get("error", {})
                except (json.JSONDecodeError, AttributeError):
                    error = {}
                error_code = error.get("code") if isinstance(error, dict) else None
                error_code = error_code if isinstance(error_code, str) else None
                retryable = exc.code in _RETRYABLE_STATUS_CODES and error_code not in {
                    "insufficient_quota", "billing_hard_limit_reached",
                }
                retry_after = headers.get("Retry-After")
                set_span_attributes(http_status=exc.code, retry_after=retry_after, error_code=error_code,
                                    retry_count=attempt, retry_exhausted=retryable and attempt >= self.cfg.max_retries,
                                    rate_limits={k.lower(): v for k, v in headers.items()
                                                 if k.lower().startswith("x-ratelimit-")})
                delay = self._rate_delay(attempt, headers) if exc.code == 429 else self._retry_delay(attempt, retry_after)
                if retryable:
                    self._cooldown(delay, reason="rate_limit" if exc.code == 429 else "server_retry",
                                   retry=attempt + 1 if attempt < self.cfg.max_retries else 0, http_status=exc.code)
                if retryable and attempt < self.cfg.max_retries:
                    attempt += 1
                    continue
                set_span_attributes(retry_count=attempt, retry_exhausted=retryable)
                raise BackendError(
                    f"OpenAI-compatible request failed for {url}: HTTP {exc.code} {exc.reason}: {body}"
                ) from exc
            except (URLError, TimeoutError, socket.timeout) as exc:
                # A refused connection means nothing is listening at all (e.g.
                # a local Ollama/LM Studio backend that was never started) --
                # retrying cannot fix that, unlike a transient network blip or
                # a slow/overloaded server, so fail fast instead of paying a
                # multi-second backoff on every call.
                connection_refused = isinstance(getattr(exc, "reason", None), ConnectionRefusedError)
                if not connection_refused and attempt < self.cfg.max_retries:
                    self._wait(self._retry_delay(attempt, None), reason="network_retry", retry=attempt + 1)
                    attempt += 1
                    continue
                set_span_attributes(retry_count=attempt, retry_exhausted=not connection_refused)
                raise BackendError(f"OpenAI-compatible request failed for {url}: {exc}") from exc

    def embeddings(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.cfg.model, "input": texts}
        response = self._post_json("/embeddings", payload)
        self.last_embedding_usage = response.get("usage") or {}
        data = response.get("data") or []
        return [list(item["embedding"]) for item in data]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings(texts)

    def chat_completion(self, messages: list[dict[str, str]], temperature: float | None = None, **extra: Any) -> str:
        payload: dict[str, Any] = {"model": self.cfg.model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        payload.update(extra)
        response = self._post_json("/chat/completions", payload)
        choices = response.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, list):
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return str(content)

    def chat_completion_with_usage(self, messages, temperature=None, **extra):
        payload: dict[str, Any] = {"model": self.cfg.model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        payload.update(extra)
        response = self._post_json("/chat/completions", payload)
        choices = response.get("choices") or []
        message = choices[0].get("message") or {} if choices else {}
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return str(content), response.get("usage") or {}

    def complete(self, messages: list[dict[str, str]], temperature: float | None = None, **extra: Any) -> str:
        return self.chat_completion(messages, temperature=temperature, **extra)

    def complete_with_usage(self, messages, temperature=None, **extra):
        return self.chat_completion_with_usage(messages, temperature=temperature, **extra)

    def chat_completion_message(self, messages: list[dict[str, Any]], temperature: float | None = None, **extra: Any):
        """Return the raw ``message`` object and ``usage`` for one chat-completion
        call, preserving fields (``tool_calls``, etc.) that the text-only helpers
        above discard. Used for OpenAI-compatible tool calling."""
        payload: dict[str, Any] = {"model": self.cfg.model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        payload.update(extra)
        response = self._post_json("/chat/completions", payload)
        choices = response.get("choices") or []
        message = choices[0].get("message") or {} if choices else {}
        return message, response.get("usage") or {}
