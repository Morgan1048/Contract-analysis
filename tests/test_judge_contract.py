from copy import deepcopy
import json

import pytest

from core.agents.judge import judge
from core.llm.stub import StubLLM
from core.pipeline import analyze
from core.retrieval.router import EvidencePack
from core.rubric.schema import SubRequirement
from eval.synthetic import rubric
from tests.test_agent_graph import make_index


def absence_payload(verdict):
    return {
        "verdict": verdict, "confidence": .8, "quote": None, "source_chunk_id": None,
        "rationale": "No obligation found in supplied evidence.", "evidence": [],
        "facet_results": [{"facet": "require a receipt", "verdict": verdict,
                          "evidence_indices": [], "rationale": "No evidence."}],
        "gaps": ["Receipt obligation missing."], "applied_modifiers": [],
    }


class PayloadClient:
    def __init__(self, payload):
        self.payload = payload

    def complete_json(self, system, user, schema, example=None):
        return self.payload, None


@pytest.mark.parametrize("verdict", ["NOT_FOUND", "UNRESOLVED"])
@pytest.mark.parametrize("facets", [[], ["require a receipt"]])
def test_null_absence_quotes_are_normalized_without_mutating_the_model_output(verdict, facets):
    payload = absence_payload(verdict)
    original = deepcopy(payload)
    finding, _ = judge(SubRequirement("receipt", "Require a receipt.", facets=facets),
                       EvidencePack("receipt", [], []), PayloadClient(payload))
    assert finding.verdict == verdict
    assert finding.quote == finding.source_chunk_id == ""
    assert finding.evidence == [] and not finding.search_complete
    assert payload == original


@pytest.mark.parametrize("field,value", [
    ("quote", []), ("source_chunk_id", 1), ("rationale", None),
])
def test_absence_does_not_coerce_other_invalid_field_types(field, value):
    payload = absence_payload("NOT_FOUND")
    payload[field] = value
    with pytest.raises(ValueError, match="must be strings"):
        judge(SubRequirement("receipt", "Require a receipt.", facets=["require a receipt"]),
              EvidencePack("receipt", [], []), PayloadClient(payload))


@pytest.mark.parametrize("verdict", ["MET", "PARTIAL", "CONTRADICTED"])
def test_supported_verdicts_cannot_use_null_as_citation(verdict):
    with pytest.raises(ValueError, match="must be strings"):
        judge(SubRequirement("receipt", "Require a receipt."),
              EvidencePack("receipt", [], []), PayloadClient(absence_payload(verdict)))


def test_null_is_not_normalized_when_evidence_is_present():
    payload = absence_payload("NOT_FOUND")
    payload["evidence"] = [{"quote": "Unverified quotation.", "source_chunk_id": "source", "role": "support"}]
    with pytest.raises(ValueError, match="must be strings"):
        judge(SubRequirement("receipt", "Require a receipt."),
              EvidencePack("receipt", [], []), PayloadClient(payload))


def test_null_absence_still_requires_alternate_search_and_critic_acceptance(tmp_path):
    class AbsenceClient(StubLLM):
        def __post_init__(self):
            super().__post_init__()
            self.routes = []

        def complete_json(self, system, user, schema, example=None):
            payload, response = super().complete_json(system, user, schema, example)
            if "ROLE: evidence_router" in system:
                self.routes.append(json.loads(user))
            if "ROLE: compliance_judge" in system and "Sub-requirement (weekly_dispatch):" in user:
                payload.update(verdict="NOT_FOUND", quote=None, source_chunk_id=None, evidence=[])
                for facet in payload["facet_results"]:
                    facet.update(verdict="NOT_FOUND", evidence_indices=[])
            return payload, response

    path, index = make_index(tmp_path)
    client = AbsenceClient()
    result = analyze(path, client, rubric={"Q1": rubric()["Q1"]}, index=index)
    assert not result.incomplete
    assert result.questions[0].state == "Partially Compliant"
    assert len(client.routes) == 2
    assert not client.routes[0]["previous_actions"]["weekly_dispatch"]
    executed = client.routes[1]["previous_actions"]["weekly_dispatch"]
    assert executed and any(action["tool"] == "search_evidence" for action in executed)
    assert not any(event["schema_repairs"] for event in result.audit_events)
    finding = next(f for f in result.questions[0].findings if f.sub_requirement_id == "weekly_dispatch")
    assert finding.verdict == "NOT_FOUND" and finding.search_complete
    assert result.usage["resolved_followup_checks"] == 1
