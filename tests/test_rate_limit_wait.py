from datetime import datetime, timedelta, timezone
from email.message import Message
from email.utils import format_datetime

import pytest

from core.agents.graph import Limits
from core.providers.config import ProviderConfig
from core.providers.errors import BackendError, RequestBudgetExceeded
from core.providers.openai_compat import OpenAICompatClient, request_deadline
from core.providers.progress import backend_wait_updates
from tests.test_live_backend_findings import _cfg, _fake_response, _http_error


@pytest.fixture
def clock(monkeypatch):
    import core.providers.openai_compat as transport

    class Clock:
        now = 100.0

        def __init__(self):
            self.sleeps = []

        def sleep(self, seconds):
            self.sleeps.append(seconds)
            self.now += seconds

    value = Clock()
    monkeypatch.setattr(transport.time, "monotonic", lambda: value.now)
    monkeypatch.setattr(transport.time, "sleep", value.sleep)
    return value


def test_wait_longer_than_old_budget_then_return_real_response(monkeypatch, clock):
    import core.providers.openai_compat as transport
    client = OpenAICompatClient(_cfg())
    attempts, updates = [], []

    def request(*_args, **_kwargs):
        attempts.append(clock.now)
        if len(attempts) == 1:
            raise _http_error(429, retry_after=600)
        return _fake_response({"choices": [{"message": {"content": "done"}}]})

    monkeypatch.setattr(transport, "urlopen", request)
    with request_deadline(Limits.from_env().seconds), backend_wait_updates(updates.append):
        assert client.chat_completion([]) == "done"
    assert attempts == [100, 700]
    assert sum(clock.sleeps) == 600
    assert updates[0].http_status == 429
    assert updates[0].remaining_s == 600
    assert updates[-1].phase == "resumed"


def test_exhausted_request_cooldown_is_shared_with_next_call(monkeypatch, clock):
    import core.providers.openai_compat as transport
    client = OpenAICompatClient(_cfg(max_retries=0))
    attempts = []

    def request(*_args, **_kwargs):
        attempts.append(clock.now)
        if len(attempts) == 1:
            raise _http_error(429, retry_after=90)
        return _fake_response({"choices": [{"message": {"content": "done"}}]})

    monkeypatch.setattr(transport, "urlopen", request)
    with pytest.raises(BackendError):
        client.chat_completion([])
    assert client.chat_completion([]) == "done"
    assert attempts == [100, 190]


def test_bounded_wait_does_not_send_more_requests(monkeypatch, clock):
    import core.providers.openai_compat as transport
    client = OpenAICompatClient(_cfg(retry_budget_s=120))
    attempts = []

    def request(*_args, **_kwargs):
        attempts.append(clock.now)
        raise _http_error(429, retry_after=600)

    monkeypatch.setattr(transport, "urlopen", request)
    with pytest.raises(RequestBudgetExceeded, match="HTTP 429.*required_wait_s=600"):
        client.chat_completion([])
    with pytest.raises(RequestBudgetExceeded):
        client.chat_completion([])
    assert attempts == [100]
    assert clock.sleeps == []


def test_rate_limit_without_retry_after_uses_minute_scale_backoff(monkeypatch, clock):
    import core.providers.openai_compat as transport
    client = OpenAICompatClient(_cfg(max_retries=2))
    monkeypatch.setattr(transport, "urlopen", lambda *_a, **_k: (_ for _ in ()).throw(_http_error(429)))
    with pytest.raises(BackendError):
        client.chat_completion([])
    assert clock.sleeps == [60, 120]


def test_http_date_and_exhausted_quota_reset_headers():
    client = OpenAICompatClient(_cfg())
    headers = Message()
    headers["Retry-After"] = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=60), usegmt=True)
    assert 58 <= client._rate_delay(0, headers) <= 60
    del headers["Retry-After"]
    headers["x-ratelimit-remaining-tokens"] = "0"
    headers["x-ratelimit-reset-tokens"] = "2m500ms"
    assert client._rate_delay(0, headers) == 120.5
    headers["Retry-After"] = "invalid"
    assert client._rate_delay(0, headers) == 120.5


def test_successful_response_exhausting_quota_delays_next_call(monkeypatch, clock):
    import core.providers.openai_compat as transport
    client = OpenAICompatClient(_cfg())
    attempts = []

    def request(*_args, **_kwargs):
        attempts.append(clock.now)
        response = _fake_response({"choices": [{"message": {"content": "ok"}}]})
        response.headers = {"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "7s"}
        return response

    monkeypatch.setattr(transport, "urlopen", request)
    client.chat_completion([])
    client.chat_completion([])
    assert attempts == [100, 107]


def test_question_and_transport_budgets_are_configurable(monkeypatch):
    monkeypatch.setenv("CA_ANALYSIS_QUESTION_TIMEOUT_S", "7200")
    monkeypatch.setenv("CA_LLM_RETRY_BUDGET_S", "3600")
    monkeypatch.setenv("CA_LLM_MAX_RETRIES", "20")
    assert Limits.from_env().seconds == 7200
    cfg = ProviderConfig.from_env("llm")
    assert cfg.retry_budget_s == 3600 and cfg.max_retries == 20
    monkeypatch.setenv("CA_ANALYSIS_QUESTION_TIMEOUT_S", "inf")
    with pytest.raises(ValueError, match="positive finite"):
        Limits.from_env()


@pytest.mark.parametrize("header", ["invalid", "inf", "nan"])
def test_invalid_retry_after_uses_rate_limit_fallback(header):
    headers = Message()
    headers["Retry-After"] = header
    assert OpenAICompatClient(_cfg())._rate_delay(0, headers) == 60


@pytest.mark.parametrize("status,code", [(401, None), (413, None), (429, "insufficient_quota"),
                                         (429, "billing_hard_limit_reached")])
def test_nonretryable_errors_do_not_wait(monkeypatch, clock, status, code):
    import io
    import json
    import core.providers.openai_compat as transport
    attempts = []

    def request(*_args, **_kwargs):
        attempts.append(clock.now)
        from urllib.error import HTTPError
        headers = Message()
        headers["Retry-After"] = "600"
        raise HTTPError("https://api.example.com/v1", status, "error", headers,
                        io.BytesIO(json.dumps({"error": {"code": code}}).encode()))

    monkeypatch.setattr(transport, "urlopen", request)
    with pytest.raises(BackendError):
        OpenAICompatClient(_cfg()).chat_completion([])
    assert len(attempts) == 1 and clock.sleeps == []
