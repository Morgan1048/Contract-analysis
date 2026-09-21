"""Tests for docs/review/11-live-backend-findings.md (X1 CRITICAL, X2 HIGH, X3 HIGH).

Each test is written to FAIL against the pre-fix code described in the review.
X1's guard test (``test_analyze_with_rate_limited_backend_is_indeterminate_not_non_compliant``)
is the single most important test in this file: before the fix, a simulated 429
produced five "Non-Compliant" questions at confidence 0.0 -- the most dangerous
failure mode this system has.
"""
from __future__ import annotations

import json
from email.message import Message
from io import BytesIO
from urllib.error import HTTPError, URLError

import certifi
import pytest

from core.agents.evaluator import judge_with_retry
from core.agents.judge import judge
from core.llm.stub import StubLLM
from core.models import Chunk, ParsedDocument
from core.pipeline import analyze
from core.providers.config import ProviderConfig
from core.providers.errors import BackendError
from core.providers.openai_compat import OpenAICompatClient
from core.retrieval.router import EvidencePack
from core.rubric.aggregate import aggregate
from core.rubric.schema import Finding, Question, SubRequirement


def _sub_and_pack():
    sub_req = SubRequirement("P1", "documented password standard", ["PASS-01"])
    chunk = Chunk("sec:6.6", "clause", "Passwords must be at least 14 characters.", 1, "6.6", None, "PASS-01", None)
    pack = EvidencePack("P1", [chunk], ["requirement_id_lookup"])
    doc = ParsedDocument("contract.pdf", 1, chunk.text, [chunk])
    return sub_req, pack, doc


# --- X1: a backend failure must become ERROR/Indeterminate, never NOT_FOUND/Non-Compliant ------


def test_judge_raises_backend_error_not_value_or_type_error_on_rate_limit():
    sub_req, pack, _doc = _sub_and_pack()
    client = StubLLM(mode="rate_limited")
    with pytest.raises(BackendError):
        judge(sub_req, pack, client)


def test_judge_converts_client_interface_typeerror_into_backend_error():
    """Reproduces the review's second failure mode: a client whose
    ``complete_json`` signature does not accept ``example`` raises a bare
    TypeError. That must surface as a BackendError, not propagate as a raw
    TypeError that a broad ``except (ValueError, TypeError)`` could catch and
    misinterpret as malformed model output.
    """
    sub_req, pack, _doc = _sub_and_pack()
    client = StubLLM(mode="interface_mismatch")
    with pytest.raises(BackendError):
        judge(sub_req, pack, client)


def test_judge_with_retry_reports_error_verdict_for_rate_limit_not_not_found():
    sub_req, pack, doc = _sub_and_pack()
    client = StubLLM(mode="rate_limited")
    finding, checks, retries = judge_with_retry(sub_req, pack, doc, client, max_retries=2)
    assert finding.verdict == "ERROR"
    assert finding.verdict != "NOT_FOUND"
    assert finding.confidence == 0.0
    assert not finding.quote_verified
    assert any(not check.passed for check in checks)


def test_judge_with_retry_reports_error_verdict_for_interface_mismatch_not_not_found():
    sub_req, pack, doc = _sub_and_pack()
    client = StubLLM(mode="interface_mismatch")
    finding, checks, retries = judge_with_retry(sub_req, pack, doc, client, max_retries=2)
    assert finding.verdict == "ERROR"


def test_judge_with_retry_does_not_retry_a_backend_error():
    """Transport-layer retries (X2) already ran; retrying the judge/verify
    loop again on top of that would only slow down an already-failed run."""
    sub_req, pack, doc = _sub_and_pack()
    client = StubLLM(mode="rate_limited")
    finding, checks, retries = judge_with_retry(sub_req, pack, doc, client, max_retries=2)
    assert retries == 0


def test_malformed_output_is_unresolved_not_absence_of_contractual_evidence():
    sub_req, pack, doc = _sub_and_pack()
    client = StubLLM(mode="malformed")
    finding, checks, retries = judge_with_retry(sub_req, pack, doc, client, max_retries=1)
    assert finding.verdict == "UNRESOLVED"
    assert not finding.quote_verified


def _question_with_error_finding():
    sub_reqs = [SubRequirement("P1", "req one", ["R1"]), SubRequirement("P2", "req two", ["R2"])]
    question = Question("Q1", "Question one", "text", sub_reqs)
    findings = [
        Finding("P1", "MET", 0.9, "quote", "sec:1", "reason", True),
        Finding("P2", "ERROR", 0.0, "", "", "Backend call failed.", False),
    ]
    return question, findings


def test_aggregate_reports_indeterminate_not_non_compliant_when_a_finding_errored():
    question, findings = _question_with_error_finding()
    result = aggregate(question, findings, [])
    assert result.state == "Indeterminate"
    assert result.state != "Non-Compliant"


def test_aggregate_error_finding_is_never_coerced_into_not_found():
    question, findings = _question_with_error_finding()
    result = aggregate(question, findings, [])
    by_id = {f.sub_requirement_id: f for f in result.findings}
    assert by_id["P2"].verdict == "ERROR"


def test_aggregate_error_question_excludes_error_findings_from_relevant_quotes():
    question, findings = _question_with_error_finding()
    result = aggregate(question, findings, [])
    assert all(f.verdict != "ERROR" for f in result.relevant_quotes)


def test_analyze_with_rate_limited_backend_is_indeterminate_not_non_compliant():
    """THE guard test: before the fix, this produced five Non-Compliant
    questions at confidence 0.0 -- a rate-limited API key presented as "the
    vendor contract has no security controls at all". After the fix, every
    question must be Indeterminate and the run must be marked incomplete.
    """
    result = analyze("Sample Contract.pdf", StubLLM(mode="rate_limited"))
    assert result.incomplete is True
    assert result.questions, "expected the rubric's questions to be present"
    for question in result.questions:
        assert question.state == "Indeterminate"
        assert question.state != "Non-Compliant"
        assert all(finding.verdict == "ERROR" for finding in question.findings)


def test_analyze_with_healthy_backend_is_complete():
    result = analyze("Sample Contract.pdf", StubLLM(mode="success"))
    assert result.incomplete is False
    assert all(question.state != "Indeterminate" for question in result.questions)


# --- X2: retry, backoff, timeout, and pacing at the transport layer -----------------------------


def _cfg(**overrides):
    defaults = dict(backend="openai", base_url="https://api.example.com/v1", model="gpt-4o-mini", api_key="secret")
    defaults.update(overrides)
    return ProviderConfig(**defaults)


def _fake_response(payload):
    class _Resp:
        def __init__(self, data):
            self._data = data

        def read(self):
            return json.dumps(self._data).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    return _Resp(payload)


def _http_error(code, retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return HTTPError("https://api.example.com/v1/chat/completions", code, "error", headers, BytesIO(b"{}"))


def test_post_json_retries_on_429_and_eventually_succeeds(monkeypatch):
    import core.providers.openai_compat as oc

    client = OpenAICompatClient(_cfg(max_retries=3, backoff_base_s=0.001, backoff_cap_s=0.001))
    calls = {"n": 0}

    def fake_urlopen(request, timeout=None, context=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _http_error(429)
        return _fake_response({"choices": [{"message": {"content": "ok"}}], "usage": {}})

    monkeypatch.setattr(oc, "urlopen", fake_urlopen)
    monkeypatch.setattr(oc.time, "sleep", lambda s: None)
    result = client.chat_completion([{"role": "user", "content": "hi"}])
    assert result == "ok"
    assert calls["n"] == 3


def test_post_json_raises_backend_error_after_retries_exhausted(monkeypatch):
    import core.providers.openai_compat as oc

    client = OpenAICompatClient(_cfg(max_retries=2, backoff_base_s=0.001, backoff_cap_s=0.001))

    def fake_urlopen(request, timeout=None, context=None):
        raise _http_error(503)

    monkeypatch.setattr(oc, "urlopen", fake_urlopen)
    monkeypatch.setattr(oc.time, "sleep", lambda s: None)
    with pytest.raises(BackendError):
        client.chat_completion([{"role": "user", "content": "hi"}])


def test_post_json_does_not_retry_non_retryable_status(monkeypatch):
    import core.providers.openai_compat as oc

    client = OpenAICompatClient(_cfg(max_retries=5))
    calls = {"n": 0}

    def fake_urlopen(request, timeout=None, context=None):
        calls["n"] += 1
        raise _http_error(401)

    monkeypatch.setattr(oc, "urlopen", fake_urlopen)
    monkeypatch.setattr(oc.time, "sleep", lambda s: None)
    with pytest.raises(BackendError):
        client.chat_completion([{"role": "user", "content": "hi"}])
    assert calls["n"] == 1


def test_post_json_does_not_retry_connection_refused(monkeypatch):
    """A refused connection (e.g. a local Ollama/LM Studio backend that was
    never started) is not a transient failure retrying can fix -- retrying it
    anyway only adds a multi-second backoff delay to every offline-first call.
    """
    import core.providers.openai_compat as oc

    client = OpenAICompatClient(_cfg(max_retries=5, backoff_base_s=1.0, backoff_cap_s=20.0))
    calls = {"n": 0}
    sleeps = []

    def fake_urlopen(request, timeout=None, context=None):
        calls["n"] += 1
        raise URLError(ConnectionRefusedError(61, "Connection refused"))

    monkeypatch.setattr(oc, "urlopen", fake_urlopen)
    monkeypatch.setattr(oc.time, "sleep", lambda s: sleeps.append(s))
    with pytest.raises(BackendError):
        client.chat_completion([{"role": "user", "content": "hi"}])
    assert calls["n"] == 1
    assert sleeps == []


def test_post_json_honours_retry_after_header(monkeypatch):
    import core.providers.openai_compat as oc

    client = OpenAICompatClient(_cfg(max_retries=2))
    sleeps = []
    calls = {"n": 0}

    def fake_urlopen(request, timeout=None, context=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _http_error(429, retry_after=7)
        return _fake_response({"choices": [{"message": {"content": "ok"}}], "usage": {}})

    monkeypatch.setattr(oc, "urlopen", fake_urlopen)
    monkeypatch.setattr(oc.time, "sleep", lambda s: sleeps.append(s))
    client.chat_completion([{"role": "user", "content": "hi"}])
    assert sleeps == pytest.approx([7.0], abs=.05)


def test_retry_after_cannot_overrun_graph_deadline(monkeypatch):
    import core.providers.openai_compat as oc
    from core.providers.errors import RequestBudgetExceeded

    client = OpenAICompatClient(_cfg(max_retries=2))
    sleeps, timeouts = [], []

    def fake_urlopen(request, timeout=None, context=None):
        timeouts.append(timeout)
        raise _http_error(429, retry_after=600)

    monkeypatch.setattr(oc, "urlopen", fake_urlopen)
    monkeypatch.setattr(oc.time, "sleep", sleeps.append)
    with oc.request_deadline(5), pytest.raises(RequestBudgetExceeded, match="retry_after"):
        client.chat_completion([{"role": "user", "content": "independent fixture"}])
    assert len(timeouts) == 1 and 0 < timeouts[0] <= 5
    assert sleeps == []
    assert oc._deadline.get() is None


def test_post_json_passes_configured_timeout(monkeypatch):
    import core.providers.openai_compat as oc

    client = OpenAICompatClient(_cfg(timeout_s=5.0))
    captured = {}

    def fake_urlopen(request, timeout=None, context=None):
        captured["timeout"] = timeout
        return _fake_response({"choices": [{"message": {"content": "ok"}}], "usage": {}})

    monkeypatch.setattr(oc, "urlopen", fake_urlopen)
    client.chat_completion([{"role": "user", "content": "hi"}])
    assert captured["timeout"] == 5.0


def test_post_json_honours_minimum_request_interval(monkeypatch):
    import core.providers.openai_compat as oc

    client = OpenAICompatClient(_cfg(min_request_interval_s=0.05))

    def fake_urlopen(request, timeout=None, context=None):
        return _fake_response({"choices": [{"message": {"content": "ok"}}], "usage": {}})

    monkeypatch.setattr(oc, "urlopen", fake_urlopen)
    sleeps = []
    monkeypatch.setattr(oc.time, "sleep", lambda s: sleeps.append(s))
    client.chat_completion([{"role": "user", "content": "hi"}])
    client.chat_completion([{"role": "user", "content": "hi"}])
    assert sleeps and sleeps[0] > 0


# --- X3: HTTPS must use certifi's CA bundle, not the platform trust store -----------------------


def test_ssl_context_is_built_from_certifi_bundle(monkeypatch):
    import ssl as ssl_module

    import core.providers.openai_compat as oc

    captured = {}
    original = ssl_module.create_default_context

    def fake_create_default_context(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(oc.ssl, "create_default_context", fake_create_default_context)
    OpenAICompatClient(_cfg())
    assert captured.get("cafile") == certifi.where()
