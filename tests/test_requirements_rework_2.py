"""Tests for docs/review/05-requirements-review.md findings R1-R8.

Each test is written to FAIL against the pre-fix code described in the review.
"""
from __future__ import annotations

import io
import json
import logging
import sys

import pytest

from core.agents.evaluator import judge_with_retry
from core.llm.stub import StubLLM
from core.models import Chunk, ParsedDocument
from core.obs.logging import get_logger
from core.output import reference_label as _reference
from core.output import to_output_document
from core.pipeline import AnalysisResult
from core.pricing import cost_usd, PRICES_USD_PER_MILLION, ZERO_RATED_USD_PER_MILLION
from core.providers.config import ProviderConfig
from core.retrieval.router import EvidencePack
from core.rubric.schema import Finding, QuestionResult, SubRequirement


# --- R1: logging must never touch stdout ------------------------------------------------

def test_logger_never_attaches_a_stdout_handler(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    logger = get_logger("test-r1-default")
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            assert handler.stream is not sys.stdout


def test_logger_stream_handler_writes_to_stderr_not_stdout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    captured_out, captured_err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured_out)
    monkeypatch.setattr(sys, "stderr", captured_err)
    logger = get_logger("test-r1-emit")
    logger.info("hello", extra={"fields": {"message": "hello"}})
    assert captured_out.getvalue() == ""
    assert "hello" in captured_err.getvalue()


def test_logger_stream_handler_is_configurable_off(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    logger = get_logger("test-r1-disabled", enable_stream=False)
    assert not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)


# --- R2: retry_count must be real, not hardcoded zero ------------------------------------

def test_finding_carries_a_retry_count_field():
    finding = Finding("P1", "MET", 0.9, "quote", "sec:1", "reason")
    assert finding.retry_count == 0


def test_judge_with_retry_populates_real_retry_count():
    sub_req = SubRequirement("P1", "documented password standard", ["PASS-01"])
    chunk = Chunk("sec:6.6", "clause", "Passwords must be at least 14 characters.", 1, "6.6", None, "PASS-01", None)
    pack = EvidencePack("P1", [chunk], ["requirement_id_lookup"])
    doc = ParsedDocument("contract.pdf", 1, chunk.text, [chunk])
    client = StubLLM(mode="invented_quote", failures_before_success=2)
    finding, checks, retries = judge_with_retry(sub_req, pack, doc, client, max_retries=2)
    assert retries == 2
    assert finding.retry_count == 2


def test_store_persists_real_retry_count_not_zero(tmp_path):
    from core.store import RunStore

    finding = Finding("P1", "MET", 0.9, "quote", "sec:1", "reason", True, retry_count=2)
    result = AnalysisResult(
        "run-retry", "contract.pdf",
        [QuestionResult("Q1", "Question", "Fully Compliant", 0.9, [finding], [], "reason")],
        {}, {"prompt_tokens": 1, "completion_tokens": 1, "cost_usd": 0.0},
        "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:01+00:00",
    )
    store = RunStore(tmp_path / "runs.db")
    store.save_run(result)
    loaded = store.get_run("run-retry")
    assert loaded.questions[0].findings[0].retry_count == 2
    row = store.db.execute("SELECT retry_count FROM findings WHERE run_id=?", ("run-retry",)).fetchone()
    assert row["retry_count"] == 2


# --- R3: unpriced (non-local) models must not silently report zero cost -----------------

def test_local_backend_costs_exactly_zero():
    assert cost_usd("llama3.1", 1000, 1000, is_local=True) == 0.0


def test_unknown_non_local_model_is_unpriced_not_zero():
    assert cost_usd("some-mystery-cloud-model", 1000, 1000, is_local=False) is None


def test_groq_free_tier_is_explicitly_zero_rated_not_unknown():
    assert "groq:openai/gpt-oss-120b" in ZERO_RATED_USD_PER_MILLION
    assert cost_usd("openai/gpt-oss-120b", 1_000_000, 1_000_000, backend="groq") == 0.0
    assert cost_usd("openai/gpt-oss-120b", 1_000_000, 1_000_000, backend=None) is None


def test_siliconflow_provider_qualified_models_are_priced():
    assert "siliconflow:openai/gpt-oss-120b" in PRICES_USD_PER_MILLION
    assert cost_usd("openai/gpt-oss-120b", 1_000_000, 1_000_000, backend="siliconflow") == pytest.approx(0.50)
    assert cost_usd("BAAI/bge-m3", 1_000_000, 0, backend="siliconflow") == pytest.approx(0.02)


def test_known_model_still_prices_correctly():
    assert cost_usd("gpt-4.1", 1_000_000, 0, is_local=False) == pytest.approx(2.0)


def test_unknown_provider_is_unpriced_even_for_known_model():
    assert cost_usd("gpt-4.1", 1_000_000, 0, provider="unknown") is None


# --- R4: embedding models must be priced -------------------------------------------------

def test_embedding_models_are_in_the_price_table():
    assert "text-embedding-3-small" in PRICES_USD_PER_MILLION
    assert cost_usd("text-embedding-3-small", 1_000_000, 0) > 0


# --- R5: the prompt's "Example" must be a real example, not the schema restated ----------

def test_complete_json_example_is_not_identical_to_schema(monkeypatch):
    from core.llm.client import LLMClient

    captured = {}
    client = LLMClient(ProviderConfig("openai", "http://example", "gpt-4o-mini"))

    def complete(messages, **kwargs):
        captured["system"] = messages[0]["content"]
        return '{"ok": true}', {"prompt_tokens": 1, "completion_tokens": 1}

    monkeypatch.setattr(client._client, "complete_with_usage", complete)
    client.complete_json("judge", "evidence", {"verdict": "string", "confidence": "number"})
    schema_line, example_line = [
        line for line in captured["system"].splitlines() if line.startswith(("Required", "Example"))
    ]
    assert schema_line.split(": ", 1)[1] != example_line.split(": ", 1)[1]
    assert '"verdict": "string"' not in example_line


@pytest.mark.parametrize("facets", [[], ["Actual facet"], ["Actual first", "Actual second"]])
def test_judge_passes_a_concrete_worked_example(facets):
    from core.agents.judge import facet_verdict, judge as judge_fn

    captured = {}

    class RecordingClient(StubLLM):
        def complete_json(self, system, user, schema, example=None):
            captured["example"] = example
            captured["system"] = system
            return super().complete_json(system, user, schema, example)

    sub_req = SubRequirement("P1", "text", facets=facets)
    chunk = Chunk("c1", "clause", "x", 1, None, None, None, None)
    pack = EvidencePack("P1", [chunk], [])
    finding, _ = judge_fn(sub_req, pack, RecordingClient())
    assert [facet["facet"] for facet in finding.facet_results] == facets
    example = captured["example"]
    assert example["evidence"]
    assert example["quote"] == example["evidence"][0]["quote"]
    assert example["source_chunk_id"] == example["evidence"][0]["source_chunk_id"]
    assert {facet["verdict"] for facet in example["facet_results"]} == {"MET", "NOT_FOUND"}
    assert example["verdict"] == facet_verdict(example["facet_results"])
    for facet in example["facet_results"]:
        indices = facet["evidence_indices"]
        assert all(type(i) is int and 0 <= i < len(example["evidence"]) for i in indices)
        assert bool(indices) == (facet["verdict"] == "MET")
    assert example["quote"] in captured["system"]
    assert example["source_chunk_id"] in captured["system"]
    assert "FICTIONAL WORKED EXAMPLE" in captured["system"]
    assert "Never copy the example" in captured["system"]
    assert "..." not in json.dumps(example)


def test_worked_example_passes_schema_and_source_checks_only_for_its_own_input():
    from core.agents.evaluator import verify
    from core.agents.judge import _worked_example, judge as judge_fn

    example = _worked_example()

    class EchoExample:
        def complete_json(self, system, user, schema, example=None):
            return example, None

    sub_req = SubRequirement("example", "Sealing and labels",
                             facets=[item["facet"] for item in example["facet_results"]])
    source = Chunk(example["source_chunk_id"], "clause", example["quote"], 1, None, None, None, None)
    pack = EvidencePack(sub_req.id, [source], [])
    finding, _ = judge_fn(sub_req, pack, EchoExample())
    doc = ParsedDocument("fictional", 1, source.text, [source])
    assert all(check.passed for check in verify(finding, pack, doc))
    actual = Chunk("actual", "clause", "An unrelated contract clause.", 1, None, None, None, None)
    actual_doc = ParsedDocument("actual", 1, actual.text, [actual])
    checks = verify(finding, EvidencePack(sub_req.id, [actual], []), actual_doc)
    assert any(check.name == "source_in_pack" and not check.passed for check in checks)
    assert any(check.name == "quote_verbatim" and not check.passed for check in checks)


# --- R6: api_key must never appear in repr() ---------------------------------------------

def test_provider_config_repr_never_shows_api_key():
    cfg = ProviderConfig("openai", "https://api.openai.com/v1", "gpt-4o-mini", api_key="sk-SECRET123")
    assert "sk-SECRET123" not in repr(cfg)
    assert "api_key" not in repr(cfg)


# --- R7: citations must carry the exhibit sub-section from the chunk heading ------------

def test_reference_includes_exhibit_sub_section_from_heading():
    chunk = Chunk(
        "exh:G:NET-01", "table_row", "TLS is required.", 4, None, "G", "NET-01",
        "G13. Network Authentication and Authorization (Added)",
    )
    assert _reference(chunk) == "Exhibit G13 (NET-01)"


def test_reference_falls_back_when_heading_has_no_sub_section():
    chunk = Chunk("exh:G:NET-01", "table_row", "TLS is required.", 4, None, "G", "NET-01", None)
    assert _reference(chunk) == "Exhibit G (NET-01)"


# --- R8: a document-level serializer must assemble all five questions -------------------

def test_to_output_document_assembles_all_questions():
    chunk = Chunk("sec:6.6", "clause", "Passwords must rotate.", 1, "6.6", None, "PASS-01", None)
    doc = ParsedDocument("contract.pdf", 1, chunk.text, [chunk])
    index = type("Index", (), {"doc": doc})()
    finding = Finding("P1", "MET", 0.9, chunk.text, chunk.chunk_id, "reason", True)
    q1 = QuestionResult("Q1", "Question one", "Fully Compliant", 0.9, [finding], [], "reason")
    q2 = QuestionResult("Q2", "Question two", "Non-Compliant", 0.0, [], [], "reason")
    result = AnalysisResult(
        "run-doc", "contract.pdf", [q1, q2], {}, {},
        "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:01+00:00",
    )
    document = to_output_document(result, index)
    assert document["run_id"] == "run-doc"
    assert len(document["questions"]) == 2
    assert document["questions"][0]["Compliance Question"] == "Question one"
    assert document["questions"][1]["Compliance State"] == "Non-Compliant"
