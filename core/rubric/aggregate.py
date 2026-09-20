from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from core.rubric.schema import Finding, Question, QuestionResult, SubRequirement

DEFAULT_RUBRIC_PATH = Path(__file__).with_name("questions.yaml")


def load_rubric(path: str | Path = DEFAULT_RUBRIC_PATH) -> dict[str, Question]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        item["id"]: Question(item["id"], item["title"], item["text"], [
            SubRequirement(sub["id"], sub["text"], facets=list(sub.get("facets", [sub["text"]])),
                           guidance=sub.get("guidance", ""))
            for sub in item["sub_requirements"]
        ]) for item in raw["questions"]
    }


def aggregate(question: Question, findings: list[Finding], deviations=()) -> QuestionResult:
    """Only verified check outcomes affect aggregation; IDs never imply overrides."""
    del deviations
    expected = {s.id for s in question.sub_requirements}
    by_id = {f.sub_requirement_id: f for f in findings}
    if len(by_id) != len(findings) or set(by_id) - expected:
        raise ValueError("duplicate or unexpected check result")
    adjusted = []
    for sub in question.sub_requirements:
        current = by_id.get(sub.id, Finding(sub.id, "UNRESOLVED", 0, "", "", "Check was not assessed.", False))
        if current.verdict != "ERROR" and not current.quote_verified:
            current = replace(current, verdict="UNRESOLVED", confidence=0,
                              rationale="Evidence verification failed. " + current.rationale)
        adjusted.append(current)
    unresolved = [f.sub_requirement_id for f in adjusted if f.verdict in {"ERROR", "UNRESOLVED"}]
    if any(f.verdict == "ERROR" for f in adjusted) or len(unresolved) == len(adjusted):
        state, confidence = "Indeterminate", 0.0
        rationale = "Analysis incomplete; unresolved checks: " + ", ".join(unresolved)
    else:
        verdicts = [f.verdict for f in adjusted]
        state = ("Fully Compliant" if verdicts and all(v == "MET" for v in verdicts) else
                 "Partially Compliant" if any(v in {"MET", "PARTIAL"} for v in verdicts) else "Non-Compliant")
        confidence = round(min((min(1.0, max(0.0, f.confidence)) for f in adjusted), default=0), 3)
        rationale = "; ".join(f"{f.sub_requirement_id}: {f.verdict} - {f.rationale}" for f in adjusted)
    modifiers = sorted({m for f in adjusted if f.quote_verified for m in f.applied_modifiers})
    return QuestionResult(question.id, question.text, state, confidence, adjusted, modifiers, rationale,
                          [f for f in adjusted if f.quote_verified and f.quote])
