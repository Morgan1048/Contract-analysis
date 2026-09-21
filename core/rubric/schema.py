from dataclasses import dataclass, field
from typing import Literal

Verdict = Literal["MET", "PARTIAL", "CONTRADICTED", "NOT_FOUND", "UNRESOLVED", "ERROR"]
ComplianceState = Literal[
    "Fully Compliant", "Partially Compliant", "Non-Compliant", "Indeterminate"
]


@dataclass(frozen=True)
class SubRequirement:
    id: str
    text: str
    requirement_ids: list[str] = field(default_factory=list)
    section_ids: list[str] = None
    facets: list[str] = field(default_factory=list)
    guidance: str = ""

    def __post_init__(self):
        if self.section_ids is None:
            object.__setattr__(self, "section_ids", [])


@dataclass(frozen=True)
class Question:
    id: str
    title: str
    text: str
    sub_requirements: list[SubRequirement]


@dataclass(frozen=True)
class Finding:
    sub_requirement_id: str
    verdict: Verdict
    confidence: float
    quote: str
    source_chunk_id: str
    rationale: str
    quote_verified: bool = True
    retry_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = 0.0
    evidence: list[dict] = field(default_factory=list)
    facet_results: list[dict] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    applied_modifiers: list[str] = field(default_factory=list)
    search_complete: bool = False


@dataclass(frozen=True)
class Deviation:
    risk_id: str
    status: str
    requirement_ids: list[str]
    text: str
    source_chunk_id: str = ""
    expiration: str | None = None


@dataclass(frozen=True)
class QuestionResult:
    question_id: str
    question_text: str
    state: ComplianceState
    confidence: float
    findings: list[Finding]
    applied_deviations: list[str]
    rationale: str
    relevant_quotes: list[object] = None

    def __post_init__(self):
        if self.relevant_quotes is None:
            object.__setattr__(self, "relevant_quotes", [])
