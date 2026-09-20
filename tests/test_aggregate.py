from dataclasses import replace

import pytest

from core.rubric.aggregate import aggregate, load_rubric
from core.rubric.schema import Deviation, Finding


Q1 = load_rubric()["Q1"]


def finding(sub_id, verdict="MET", confidence=0.9, verified=True):
    return Finding(sub_id, verdict, confidence, "evidence", f"sec:{sub_id}", "reason", verified)


def q1_findings(verdict="MET"):
    return [finding(sub.id, verdict) for sub in Q1.sub_requirements]


def deviation(status="Open", ids=None):
    return Deviation("RA-2026-001", status, ids or ["PASS-02"], "risk")


def test_all_met_is_fully_compliant():
    assert aggregate(Q1, q1_findings(), []).state == "Fully Compliant"


def test_shared_control_id_does_not_automatically_downgrade_findings():
    result = aggregate(Q1, q1_findings(), [deviation()])
    assert result.state == "Fully Compliant"
    assert result.applied_deviations == []
    assert [f.verdict for f in result.findings] == ["MET"] * 7


def test_closed_deviation_does_not_apply():
    result = aggregate(Q1, q1_findings(), [deviation("Closed")])
    assert result.state == "Fully Compliant"
    assert result.applied_deviations == []


def test_verified_applicable_modifier_is_preserved():
    findings = q1_findings()
    findings[0] = replace(findings[0], verdict="PARTIAL", applied_modifiers=["source:exception"])
    result = aggregate(Q1, findings, [])
    assert result.state == "Partially Compliant"
    assert result.applied_deviations == ["source:exception"]


def test_unmapped_deviation_does_not_apply():
    result = aggregate(Q1, q1_findings(), [deviation(ids=["LOG-02"])])
    assert result.state == "Fully Compliant"
    assert result.applied_deviations == []


def test_all_not_found_is_non_compliant():
    assert aggregate(Q1, q1_findings("NOT_FOUND"), []).state == "Non-Compliant"


def test_met_and_not_found_is_partial():
    findings = q1_findings()
    findings[-1] = replace(findings[-1], verdict="NOT_FOUND")
    assert aggregate(Q1, findings, []).state == "Partially Compliant"


def test_unverified_met_requires_review():
    findings = q1_findings()
    findings[0] = replace(findings[0], quote_verified=False)
    result = aggregate(Q1, findings, [])
    assert result.findings[0].verdict == "UNRESOLVED"
    assert result.findings[0].confidence == 0.0
    assert result.state == "Partially Compliant"
    assert result.confidence == 0.0
    assert f"{findings[0].sub_requirement_id}: UNRESOLVED" in result.rationale
    assert "Evidence verification failed" in result.rationale


@pytest.mark.parametrize(("resolved_verdict", "expected_state"), [
    ("MET", "Partially Compliant"),
    ("PARTIAL", "Partially Compliant"),
    ("CONTRADICTED", "Non-Compliant"),
    ("NOT_FOUND", "Non-Compliant"),
])
def test_unresolved_check_does_not_delete_resolved_question(resolved_verdict, expected_state):
    findings = q1_findings(resolved_verdict)
    findings[-1] = replace(findings[-1], verdict="UNRESOLVED", confidence=0,
                           quote_verified=False, rationale="Required attachment is unavailable.")
    result = aggregate(Q1, findings)
    assert result.state == expected_state
    assert result.confidence == 0.0
    assert f"{findings[-1].sub_requirement_id}: UNRESOLVED" in result.rationale
    assert "Required attachment is unavailable." in result.rationale
    assert len(result.findings) == len(Q1.sub_requirements)


def test_missing_check_is_unresolved_without_deleting_question():
    result = aggregate(Q1, q1_findings()[1:])
    assert result.state == "Partially Compliant"
    assert result.confidence == 0.0
    assert result.findings[0].verdict == "UNRESOLVED"
    assert f"{Q1.sub_requirements[0].id}: UNRESOLVED" in result.rationale


@pytest.mark.parametrize("findings", [
    [],
    q1_findings("UNRESOLVED"),
    [replace(f, quote_verified=False) for f in q1_findings()],
], ids=["missing", "unresolved", "unverified"])
def test_no_resolved_checks_requires_review(findings):
    result = aggregate(Q1, findings)
    assert result.state == "Indeterminate"
    assert result.confidence == 0.0
    assert all(f.verdict == "UNRESOLVED" for f in result.findings)


def test_backend_error_keeps_question_indeterminate():
    findings = q1_findings()
    findings[-1] = replace(findings[-1], verdict="ERROR", confidence=0,
                           quote_verified=False, rationale="Backend unavailable.")
    result = aggregate(Q1, findings)
    assert result.state == "Indeterminate"
    assert result.confidence == 0.0
    assert result.findings[-1].verdict == "ERROR"
    assert findings[-1].sub_requirement_id in result.rationale


def test_fully_compliant_confidence_uses_minimum():
    findings = [finding(sub.id, confidence=i / 10) for i, sub in enumerate(Q1.sub_requirements, 5)]
    assert aggregate(Q1, findings, []).confidence == 0.5


def test_partial_confidence_does_not_average_away_weak_evidence():
    findings = [finding(sub.id, confidence=i / 10) for i, sub in enumerate(Q1.sub_requirements, 5)]
    result = aggregate(Q1, findings, [deviation()])
    assert result.confidence == 0.5


def test_non_compliant_confidence_uses_minimum():
    findings = [finding(sub.id, "NOT_FOUND", i / 10) for i, sub in enumerate(Q1.sub_requirements, 5)]
    assert aggregate(Q1, findings, []).confidence == 0.5


def test_deviation_does_not_change_not_found():
    findings = q1_findings()
    findings[5] = replace(findings[5], verdict="NOT_FOUND")
    result = aggregate(Q1, findings, [deviation()])
    assert result.findings[5].verdict == "NOT_FOUND"


def test_aggregate_does_not_mutate_inputs():
    findings = q1_findings()
    original = list(findings)
    aggregate(Q1, findings, [deviation()])
    assert findings == original
