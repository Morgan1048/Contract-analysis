from copy import deepcopy

import pytest

from core.agents.graph import Limits, _critic_reviews
from core.llm.stub import StubLLM
from core.pipeline import analyze
from eval.synthetic import rubric
from tests.test_agent_graph import make_index


@pytest.mark.parametrize("gap", [None, "", "Supported by the supplied evidence."])
def test_accept_allows_no_gap_without_mutating_model_payload(gap):
    payload = {"reviews": [{"check_id": "check-1", "decision": "accept", "confidence": .9, "gap": gap}]}
    original = deepcopy(payload)
    assert _critic_reviews(payload, ["check-1"])[0]["gap"] == (gap or "")
    assert payload == original


def test_accept_allows_omitted_gap():
    payload = {"reviews": [{"check_id": "check-1", "decision": "accept", "confidence": .9}]}
    assert _critic_reviews(payload, ["check-1"])[0]["gap"] == ""


@pytest.mark.parametrize("decision", ["retrieve", "unresolved"])
@pytest.mark.parametrize("gap", [None, "", " \n", [], {}])
def test_non_acceptance_still_requires_actionable_string(decision, gap):
    payload = {"reviews": [{"check_id": "check-1", "decision": decision, "confidence": .8, "gap": gap}]}
    with pytest.raises(ValueError):
        _critic_reviews(payload, ["check-1"])


@pytest.mark.parametrize("gap", [[], {}, 0, False])
def test_accept_does_not_coerce_invalid_gap_types(gap):
    payload = {"reviews": [{"check_id": "check-1", "decision": "accept", "confidence": .9, "gap": gap}]}
    with pytest.raises(ValueError, match="must be a string"):
        _critic_reviews(payload, ["check-1"])


@pytest.mark.parametrize("confidence", [None, True, -1, 1.1, float("nan"), "0.9"])
def test_nullable_gap_does_not_weaken_confidence_validation(confidence):
    payload = {"reviews": [{"check_id": "check-1", "decision": "accept", "confidence": confidence, "gap": None}]}
    with pytest.raises(ValueError):
        _critic_reviews(payload, ["check-1"])


class NullableGapClient(StubLLM):
    omit_gap = False
    invalid_quotes = False

    def complete_json(self, system, user, schema, example=None):
        payload, response = super().complete_json(system, user, schema, example)
        if "ROLE: evidence_critic" in system:
            for review in payload["reviews"]:
                if self.omit_gap:
                    review.pop("gap", None)
                else:
                    review["gap"] = None
        if self.invalid_quotes and "ROLE: compliance_judge" in system:
            payload["evidence"][0]["quote"] = "This fabricated sentence does not occur in the contract."
        return payload, response


@pytest.mark.parametrize("omit_gap", [False, True])
def test_null_or_missing_acceptance_gap_needs_no_extra_model_calls(tmp_path, omit_gap):
    path, index = make_index(tmp_path)
    client = NullableGapClient()
    client.omit_gap = omit_gap
    result = analyze(path, client, rubric={"Q1": rubric()["Q1"]}, index=index)
    assert not result.incomplete
    assert result.usage["llm_calls"] == 7  # Router + five judges + critic.
    assert all(f.quote_verified and f.search_complete for f in result.questions[0].findings)
    assert all(not e["schema_repairs"] and e["error"] is None for e in result.audit_events)
    assert result.audit_events[-1]["stop_reason"] == "complete"


def test_nullable_acceptance_never_overrides_invalid_quotes(tmp_path):
    path, index = make_index(tmp_path)
    client = NullableGapClient()
    client.invalid_quotes = True
    result = analyze(path, client, rubric={"Q1": rubric()["Q1"]}, index=index,
                     limits=Limits(followups=0))
    assert result.incomplete
    assert result.audit_events[-1]["stop_reason"] == "followup_budget"
    assert not result.audit_events[-1]["schema_repairs"]
    assert all(not f.quote_verified for f in result.questions[0].findings)
    assert all(any(c.name == "quote_verbatim" and not c.passed for c in checks)
               for checks in result.checks.values())
