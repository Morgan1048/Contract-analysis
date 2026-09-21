from copy import deepcopy
from dataclasses import replace
import json

import pytest

from core.agents.evaluator import _expand_elision, restore_elided_quotes, verify
from core.agents.graph import Limits
from core.llm.stub import StubLLM
from core.llm.client import PromptBudgetExceeded
from core.models import Chunk, ParsedDocument
from core.output import to_output_document
from core.pipeline import analyze
from core.providers.errors import BackendError
from core.retrieval.router import EvidencePack
from core.rubric.schema import Finding
from core.store import RunStore
from eval.synthetic import rubric
from tests.test_agent_graph import make_index


@pytest.mark.parametrize("marker", ["...", "\u2026"])
def test_expansion_restores_all_intervening_words_including_negations(marker):
    source = "Supplier shall not dispatch cartons unless approved, and must retain the receipt."
    quote = f"Supplier shall {marker} dispatch cartons {marker} retain the receipt."
    assert _expand_elision(quote, source) == source


@pytest.mark.parametrize("quote,source", [
    ("Supplier shall ... retain receipts.", "Supplier must seal cartons and retain receipts."),
    ("supplier shall ... retain receipts.", "Supplier shall seal cartons and retain receipts."),
    ("must not ... dispatch.", "Supplier mustnot allow dispatch."),
    ("end ... start", "start middle end"),
    ("start ... end", "start middle start middle end"),
    ("start ... end", "start middle end and end"),
    ("start ... end", "start middle finish"),
    ("start ... ", "start middle end"),
    ("... end", "start middle end"),
    ("...", "start middle end"),
    ("start ... end", "start ... end"),
    ("a b ... b c", "a b c"),
    ("start ... nonexistent ... end", "start middle end"),
])
def test_ambiguous_missing_reordered_or_already_literal_elisions_are_not_changed(quote, source):
    assert _expand_elision(quote, source) is None


def test_only_whitespace_is_normalized():
    source = "Supplier shall\nseal cartons;\tretain receipts."
    assert _expand_elision("Supplier  shall ... retain receipts.", source) == "Supplier shall seal cartons; retain receipts."


def fixture():
    text = "Supplier shall seal cartons, including: (a) boxes; (b) crates; (c) returnable containers."
    chunk = Chunk("source", "clause", text, 1, None, None, None, None)
    quote = "Supplier shall seal cartons, including: ... (c) returnable containers."
    finding = Finding(
        "sealing", "MET", .9, quote, chunk.chunk_id, "Claim to be reviewed.",
        evidence=[
            {"quote": "(a) boxes;", "source_chunk_id": chunk.chunk_id, "role": "support"},
            {"quote": quote, "source_chunk_id": chunk.chunk_id, "role": "condition"},
        ],
        facet_results=[{"facet": "sealing", "verdict": "MET", "evidence_indices": [0, 1], "rationale": "Claim."}],
    )
    return finding, EvidencePack("sealing", [chunk], []), ParsedDocument("synthetic", 1, text, [chunk])


def test_restoration_preserves_indices_roles_verdict_and_original_model_output():
    finding, pack, doc = fixture()
    original = deepcopy(finding)
    assert any(not check.passed for check in verify(finding, pack, doc))
    restored, changes = restore_elided_quotes(finding, pack)
    assert finding == original
    assert len(restored.evidence) == 2
    assert restored.evidence[0] == finding.evidence[0]
    assert restored.evidence[1]["role"] == "condition"
    assert restored.evidence[1]["quote"] == doc.text
    assert restored.facet_results == finding.facet_results
    assert restored.verdict == finding.verdict and restored.confidence == finding.confidence
    assert not restored.quote_verified and not restored.search_complete
    assert changes == [{
        "check_id": finding.sub_requirement_id, "evidence_index": 1, "source_chunk_id": "source",
        "original_quote": finding.evidence[1]["quote"], "restored_quote": doc.text,
    }]
    assert all(check.passed for check in verify(restored, pack, doc))


def test_no_cross_source_matching_or_silent_deletion_of_bad_evidence():
    finding, pack, doc = fixture()
    wrong = replace(finding, evidence=[{**finding.evidence[1], "source_chunk_id": "not-supplied"}])
    assert restore_elided_quotes(wrong, pack) == (wrong, [])
    mixed = replace(finding, evidence=[*finding.evidence, {
        "quote": "An invented sentence.", "source_chunk_id": "source", "role": "counter",
    }])
    restored, changes = restore_elided_quotes(mixed, pack)
    assert len(changes) == 1 and len(restored.evidence) == 3
    assert restored.evidence[-1] == mixed.evidence[-1]
    assert any(not check.passed for check in verify(restored, pack, doc))


def test_legacy_flat_quote_retains_its_shape_and_must_be_reverified():
    finding, pack, doc = fixture()
    flat = replace(finding, evidence=[], facet_results=[])
    restored, changes = restore_elided_quotes(flat, pack)
    assert changes and restored.quote == doc.text
    assert restored.evidence == []
    assert all(check.passed for check in verify(restored, pack, doc))


@pytest.mark.parametrize("accept", [False, True])
def test_restored_quotes_are_audited_and_require_fresh_critic_review(tmp_path, accept):
    class ElisionClient(StubLLM):
        def complete_json(self, system, user, schema, example=None):
            payload, response = super().complete_json(system, user, schema, example)
            if "ROLE: compliance_judge" in system and "Sub-requirement (weekly_dispatch):" in user:
                words = payload["evidence"][0]["quote"].split()
                payload["evidence"][0]["quote"] = " ".join(words[:4]) + " ... " + " ".join(words[-4:])
            if "ROLE: evidence_critic" in system:
                request = json.loads(user)
                assert "especially negations" in system
                change = request["quote_restorations"]["weekly_dispatch"][0]
                finding = next(f for f in request["findings"] if f["sub_requirement_id"] == "weekly_dispatch")
                assert not finding["quote_verified"]
                assert "pending YOUR acceptance" in system
                assert finding["evidence"][0]["quote"] == change["restored_quote"]
                assert change["original_quote"] != change["restored_quote"]
                if not accept:
                    payload["reviews"][0].update(decision="unresolved", gap="Restored conditions do not support the verdict.")
            return payload, response

    path, index = make_index(tmp_path)
    result = analyze(path, ElisionClient(), rubric={"Q1": rubric()["Q1"]}, index=index,
                     limits=Limits(followups=0))
    assert result.incomplete is not accept
    assert result.usage["llm_calls"] == 7
    assert result.usage["quote_attempts"] == 6
    assert result.usage["quote_failures"] == 1
    assert result.usage["followup_checks"] == 0
    finding = next(f for f in result.questions[0].findings if f.sub_requirement_id == "weekly_dispatch")
    assert finding.quote_verified is accept
    gate = next(c for c in result.checks["weekly_dispatch"] if c.name == "restoration_review")
    assert gate.passed is accept
    event = next(e for e in result.audit_events if e["node"] == "verify")
    assert len(event["quote_restorations"]) == 1
    assert event["llm_calls"] == 0
    store = RunStore(tmp_path / "runs.db")
    try:
        store.save_run(result)
        assert store.get_run(result.run_id) == result
    finally:
        store.db.close()


@pytest.mark.parametrize("failure", ["call_budget", "backend", "invalid_output", "prompt_budget", "partial_batch"])
def test_restoration_cannot_escape_when_critic_fails(tmp_path, failure):
    class FailingCritic(StubLLM):
        def complete_json(self, system, user, schema, example=None):
            if "ROLE: evidence_critic" in system:
                if failure == "backend":
                    raise BackendError("Critic unavailable")
                if failure == "prompt_budget":
                    raise PromptBudgetExceeded("prompt_token_budget")
                if failure == "partial_batch":
                    ids = json.loads(user)["pending_check_ids"]
                    if len(ids) > 1:
                        raise PromptBudgetExceeded("prompt_token_budget")
                    if ids != ["weekly_dispatch"]:
                        raise BackendError("Later critic batch failed")
            payload, response = super().complete_json(system, user, schema, example)
            if "ROLE: compliance_judge" in system and "Sub-requirement (weekly_dispatch):" in user:
                item = payload["evidence"][0]
                words = item["quote"].split()
                item["quote"] = " ".join(words[2:6]) + " ... " + " ".join(words[-4:])
            if "ROLE: evidence_critic" in system and failure == "invalid_output":
                payload = {"reviews": []}
            return payload, response

    path, index = make_index(tmp_path)
    result = analyze(path, FailingCritic(), rubric={"Q1": rubric()["Q1"]}, index=index,
                     limits=Limits(llm_calls=6 if failure == "call_budget" else 32))
    assert result.incomplete
    finding = next(f for f in result.questions[0].findings if f.sub_requirement_id == "weekly_dispatch")
    assert finding.verdict in {"ERROR", "UNRESOLVED"}
    assert not finding.quote_verified and not finding.search_complete
    assert finding.confidence == 0
    assert any(c.name == "restoration_review" and not c.passed for c in result.checks["weekly_dispatch"])
    assert any(c.name == "quote_verbatim" and c.passed for c in result.checks["weekly_dispatch"])
    store = RunStore(tmp_path / "failed-critic.db")
    try:
        store.save_run(result)
        loaded = store.get_run(result.run_id)
        assert loaded == result
        output = to_output_document(loaded, None)
        assert output["status"] == "incomplete"
        assert all(item["quote"] != finding.quote
                   for q in output["questions"] for item in q["Relevant Quotes"])
    finally:
        store.db.close()


@pytest.mark.parametrize("flat_source", ["", "unrelated-source"])
def test_restoration_synchronizes_legacy_quote_source_pair(tmp_path, flat_source):
    from core.rubric.schema import QuestionResult
    from core.pipeline import AnalysisResult

    finding, pack, doc = fixture()
    other = replace(pack.chunks[0], chunk_id="first-source", text="Supplier shall provide receipts.")
    pack.chunks.append(other)
    finding = replace(finding, quote="", source_chunk_id=flat_source, evidence=[
        {"quote": other.text, "source_chunk_id": other.chunk_id, "role": "support"},
        finding.evidence[1],
    ])
    restored, changes = restore_elided_quotes(finding, pack)
    assert changes and (restored.quote, restored.source_chunk_id) == (other.text, other.chunk_id)
    doc = replace(doc, text=f"{doc.text} {other.text}", chunks=[*doc.chunks, other])
    assert all(check.passed for check in verify(restored, pack, doc))
    # Simulate downstream acceptance, then check both persistence representations.
    accepted = replace(restored, quote_verified=True, search_complete=True)
    result = AnalysisResult(
        "paired-citation", "synthetic.pdf",
        [QuestionResult("Q1", "Question", "Fully Compliant", .9, [accepted], [], "Reviewed.")],
        {}, {}, "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:01+00:00",
    )
    store = RunStore(tmp_path / "paired.db")
    try:
        store.save_run(result)
        assert store.get_run(result.run_id).questions[0].findings[0].evidence == accepted.evidence
        row = store.db.execute("SELECT quote, source_chunk_id FROM findings").fetchone()
        assert (row["quote"], row["source_chunk_id"]) == (other.text, other.chunk_id)
        store.db.execute("DELETE FROM run_snapshots WHERE run_id=?", (result.run_id,))
        legacy = to_output_document(store.get_run(result.run_id), None)
        assert legacy["questions"][0]["Relevant Quotes"][0]["source_chunk_id"] == other.chunk_id
    finally:
        store.db.close()
