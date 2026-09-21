from dataclasses import replace
import json

import pytest

from core.agents.graph import Limits
from core.agents.evaluator import verify
from core.llm.client import LLMClient, PromptBudgetExceeded
from core.llm.stub import StubLLM
from core.pipeline import analyze
from core.providers.config import ProviderConfig
from core.models import Chunk, ParsedDocument
from core.retrieval.router import EvidencePack
from core.rubric.schema import Finding
from core.store import RunStore
from eval.synthetic import rubric
from tests.test_agent_graph import make_index


class SplicedQuoteClient(StubLLM):
    repair_mode = "success"
    reject_semantics = False
    target = "weekly_dispatch"

    def __post_init__(self):
        super().__post_init__()
        self.repairs = []
        self.initial_evidence = None
        self.critic_inputs = []

    def complete_json(self, system, user, schema, example=None):
        payload, response = super().complete_json(system, user, schema, example)
        if "ROLE: compliance_judge" in system:
            assert "one continuous, verbatim substring" in system
            assert "separate evidence entries" in system
            assert "governing clause" in system
            assert "..." not in json.dumps(example)
            if f"Sub-requirement ({self.target}):" in user:
                item = payload["evidence"][0]
                words = item["quote"].split()
                quotes = [" ".join(words[:4]), " ".join(words[-4:])]
                source_text = user.split("\nEvidence:\n", 1)[1]
                if "QUOTATION REPAIR:" in user:
                    self.repairs.append(user)
                    assert source_text == self.initial_evidence
                    assert "quote is not in the supplied chunk" in user
                    assert "previous_evidence" in user
                    assert "update facet evidence_indices" in user
                    if self.repair_mode == "malformed":
                        payload["verdict"] = "INVALID"
                        return payload, response
                    if self.repair_mode == "success":
                        payload["evidence"] = [{**item, "quote": quote} for quote in quotes]
                        for facet in payload["facet_results"]:
                            facet["evidence_indices"] = [0, 1]
                        payload["quote"] = quotes[0]
                        return payload, response
                else:
                    self.initial_evidence = source_text
                payload["evidence"][0]["quote"] = "Incorrect wording " + " ... ".join(quotes)
        if "ROLE: evidence_critic" in system:
            self.critic_inputs.append(json.loads(user))
            if self.reject_semantics:
                for review in payload["reviews"]:
                    if review["check_id"] == self.target:
                        review.update(decision="unresolved", gap="The governing obligation is ambiguous.")
        return payload, response


def run_question(tmp_path, client, **kwargs):
    path, index = make_index(tmp_path)
    return analyze(path, client, rubric={"Q1": rubric()["Q1"]}, index=index, **kwargs)


def test_nonrestorable_quote_is_repaired_once_without_new_retrieval(tmp_path):
    client = SplicedQuoteClient()
    result = run_question(tmp_path, client, limits=Limits(followups=0))
    assert not result.incomplete
    assert len(client.repairs) == 1
    assert [e["node"] for e in result.audit_events] == [
        "router", "retrieve", "judge", "verify", "quote_repair", "critic",
    ]
    assert result.usage["llm_calls"] == 8  # Router, five judges, one correction, critic.
    assert result.usage["quote_attempts"] == 6
    assert result.usage["quote_failures"] == 1
    assert result.usage["followup_checks"] == 0
    assert result.usage["unaccounted_llm_calls"] == 0
    repair = next(e for e in result.audit_events if e["node"] == "quote_repair")
    assert repair["llm_calls"] == 1 and not repair["tool_calls"]
    assert [r["check_id"] for r in repair["quote_repairs"]] == [client.target]
    assert repair["quote_attempts"] == [{"check_id": client.target, "passed": True}]
    finding = next(f for f in result.questions[0].findings if f.sub_requirement_id == client.target)
    assert finding.quote_verified and finding.search_complete and finding.retry_count == 1
    assert len(finding.evidence) == 2
    assert all(f["evidence_indices"] == [0, 1] for f in finding.facet_results)
    reviewed = next(f for f in client.critic_inputs[0]["findings"] if f["sub_requirement_id"] == client.target)
    assert reviewed["evidence"] == finding.evidence
    assert all(c.passed for c in result.checks[client.target])
    judge_event = next(e for e in result.audit_events if e["node"] == "judge")
    findings = result.questions[0].findings
    assert sum(f.prompt_tokens for f in findings) == judge_event["prompt_tokens"] + repair["prompt_tokens"]
    assert sum(f.completion_tokens for f in findings) == judge_event["completion_tokens"] + repair["completion_tokens"]
    store = RunStore(tmp_path / "repairs.db")
    try:
        store.save_run(result)
        assert store.get_run(result.run_id) == result
    finally:
        store.db.close()


@pytest.mark.parametrize("followups", [0, 2])
def test_persistent_nonrestorable_quote_stays_unverified_with_only_one_repair(tmp_path, followups):
    client = SplicedQuoteClient()
    client.repair_mode = "unchanged"
    result = run_question(tmp_path, client, limits=Limits(followups=followups))
    assert result.incomplete
    assert len(client.repairs) == 1
    assert sum(len(e["quote_repairs"]) for e in result.audit_events) == 1
    finding = next(f for f in result.questions[0].findings if f.sub_requirement_id == client.target)
    assert finding.verdict == "UNRESOLVED"
    assert not finding.quote_verified and not finding.search_complete
    assert any(c.name == "quote_verbatim" and not c.passed for c in result.checks[client.target])
    assert all(f.search_complete for f in result.questions[0].findings if f.sub_requirement_id != client.target)


def test_quote_correction_does_not_bypass_semantic_critic(tmp_path):
    client = SplicedQuoteClient()
    client.reject_semantics = True
    result = run_question(tmp_path, client)
    assert result.incomplete
    finding = next(f for f in result.questions[0].findings if f.sub_requirement_id == client.target)
    assert finding.quote_verified
    assert not finding.search_complete
    assert finding.verdict == "UNRESOLVED"
    assert "governing obligation" in finding.rationale


def test_malformed_correction_does_not_start_unbounded_schema_repairs(tmp_path):
    client = SplicedQuoteClient()
    client.repair_mode = "malformed"
    result = run_question(tmp_path, client, limits=Limits(followups=0))
    assert result.incomplete
    assert len(client.repairs) == 1
    event = next(e for e in result.audit_events if e["node"] == "quote_repair")
    assert event["llm_calls"] == 1
    assert len(event["schema_repairs"]) == 1
    finding = next(f for f in result.questions[0].findings if f.sub_requirement_id == client.target)
    assert finding.verdict == "UNRESOLVED" and not finding.search_complete


def test_quote_repair_respects_call_budget_and_preserves_verified_findings(tmp_path):
    client = SplicedQuoteClient()
    result = run_question(tmp_path, client, limits=Limits(llm_calls=6))
    assert result.incomplete and result.usage["budget_exhausted"]
    assert result.usage["llm_calls"] == 6
    assert not client.repairs
    assert result.audit_events[-1]["stop_reason"] == "llm_call_budget"
    unaffected = [f for f in result.questions[0].findings if f.sub_requirement_id != client.target]
    assert all(f.verdict == "MET" and f.quote_verified for f in unaffected)


def test_valid_quotes_do_not_use_the_repair_path(tmp_path):
    result = run_question(tmp_path, StubLLM(), limits=Limits(followups=0))
    assert not result.incomplete
    assert result.usage["llm_calls"] == 7
    assert result.usage["quote_attempts"] == 5
    assert not any(e["quote_repairs"] for e in result.audit_events)


def test_exact_elisions_are_restored_without_additional_model_calls(tmp_path):
    class AllSplicedClient(StubLLM):
        def complete_json(self, system, user, schema, example=None):
            payload, response = super().complete_json(system, user, schema, example)
            if "ROLE: compliance_judge" in system and "QUOTATION REPAIR:" not in user:
                item = payload["evidence"][0]
                words = item["quote"].split()
                item["quote"] = " ".join(words[:4]) + " ... " + " ".join(words[-4:])
            return payload, response

    result = run_question(tmp_path, AllSplicedClient(), limits=Limits(followups=0))
    assert not result.incomplete
    assert result.usage["llm_calls"] == 7
    assert result.usage["quote_attempts"] == 10
    assert result.usage["quote_failures"] == 5
    assert all(f.quote_verified and f.search_complete for f in result.questions[0].findings)
    verification = next(e for e in result.audit_events if e["node"] == "verify")
    assert len(verification["quote_restorations"]) == 5
    assert verification["llm_calls"] == 0
    assert not any(e["node"] == "quote_repair" for e in result.audit_events)


def test_literal_source_ellipsis_is_still_valid():
    text = "The printed label must read: Delivery ... pending."
    chunk = Chunk("label", "clause", text, 1, None, None, None, None)
    pack = EvidencePack("labels", [chunk], [])
    doc = ParsedDocument("synthetic", 1, text, [chunk])
    finding = Finding("labels", "MET", .9, "Delivery ... pending.", "label", "Label wording.")
    assert all(check.passed for check in verify(finding, pack, doc))


def test_higher_input_limit_is_explicit_and_still_enforced(monkeypatch):
    monkeypatch.setenv("CA_LLM_MAX_INPUT_TOKENS", "32768")
    cfg = ProviderConfig.from_env("llm")
    cfg = replace(cfg, model="gemma4:31b")
    client = LLMClient(cfg)
    system, user, schema = "Return JSON.", "hello " * 5600, {"value": "string"}
    count = client.validate_prompt(system, user, schema)
    assert 5500 < count < 32768
    with pytest.raises(PromptBudgetExceeded):
        LLMClient(replace(cfg, max_input_tokens=5500)).validate_prompt(system, user, schema)
    with pytest.raises(PromptBudgetExceeded):
        client.validate_prompt(system, "hello " * 33000, schema)
