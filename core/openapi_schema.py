"""OpenAPI schemas derived from the public pipeline dataclasses."""
from __future__ import annotations

from dataclasses import fields

from core.models import ParsedDocument
from core.pipeline import AnalysisResult
from core.rubric.schema import QuestionResult


def compliance_question_schema() -> dict:
    field_names = {field.name for field in fields(QuestionResult)}
    assert {"question_text", "state", "confidence", "rationale"} <= field_names
    return {
        "type": "object",
        "required": ["Compliance Question", "Compliance State", "Confidence", "Relevant Quotes", "Rationale"],
        "properties": {
            "Compliance Question": {"type": "string"},
            "Compliance State": {"type": "string", "enum": ["Fully Compliant", "Partially Compliant", "Non-Compliant"]},
            "Confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "Relevant Quotes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["quote", "citation", "source_chunk_id"],
                    "properties": {
                        "quote": {"type": "string"},
                        "citation": {"type": "string"},
                        "source_chunk_id": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
            "Rationale": {"type": "string"},
        },
        "additionalProperties": False,
    }


def analysis_result_schema() -> dict:
    field_names = {field.name for field in fields(AnalysisResult)}
    assert {"run_id", "source_path", "started_at", "finished_at", "questions", "incomplete"} <= field_names
    return {
        "type": "object",
        "required": ["run_id", "source_path", "started_at", "finished_at", "status", "questions"],
        "properties": {
            "run_id": {"type": "string"},
            "source_path": {"type": "string"},
            "started_at": {"type": "string", "format": "date-time"},
            "finished_at": {"type": "string", "format": "date-time"},
            # X1: mirrors AnalysisResult.incomplete -- "incomplete" when a
            # backend failure prevented one or more sub-requirements from
            # being judged, "completed" otherwise.
            "status": {"type": "string", "enum": ["completed", "incomplete"]},
            "questions": {"type": "array", "maxItems": 5, "items": compliance_question_schema()},
            "review_required": {"type": "array", "items": {"type": "object"}},
            "check_results": {"type": "array", "items": {"type": "object"}},
            "audit": {"type": "object"},
        },
        "additionalProperties": False,
    }
