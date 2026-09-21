import json
from pathlib import Path

from core.ingest.parse import parse_pdf
from core.llm.client import LLMClient
from core.models import Chunk, ParsedDocument
from core.output import to_compliance_json
from core.providers.config import ProviderConfig
from core.rubric.schema import Finding, QuestionResult


def test_output_has_assignment_fields_and_human_citations():
    chunk = Chunk("exh:G:NET-01", "table_row", "TLS is required.", 4, None, "G", "NET-01", None)
    doc = ParsedDocument("contract.pdf", 1, chunk.text, [chunk])
    result = QuestionResult("Q5", "Network question", "Fully Compliant", 0.9,
                            [Finding("N1", "MET", 0.9, chunk.text, chunk.chunk_id, "reason", True)],
                            [], "reason")
    rendered = to_compliance_json(result, type("Index", (), {"doc": doc})())
    assert set(rendered) == {"Compliance Question", "Compliance State", "Confidence", "Relevant Quotes", "Rationale"}
    assert rendered["Relevant Quotes"][0]["citation"] == "Exhibit G (NET-01)"


def test_unverified_findings_are_not_rendered():
    finding = Finding("N1", "MET", 0.9, "invented", "missing", "reason", False)
    result = QuestionResult("Q5", "Network question", "Non-Compliant", 0.0, [finding], [], "reason")
    doc = ParsedDocument("contract.pdf", 1, "", [])
    assert to_compliance_json(result, type("Index", (), {"doc": doc})())["Relevant Quotes"] == []


def test_llm_prompt_includes_requested_schema_and_provider_usage(monkeypatch):
    captured = {}
    client = LLMClient(ProviderConfig("openai", "https://api.openai.com/v1", "gpt-4o-mini"))

    def complete(messages, **kwargs):
        captured["system"] = messages[0]["content"]
        return '{"ok": true}', {"prompt_tokens": 12, "completion_tokens": 4}

    monkeypatch.setattr(client._client, "complete_with_usage", complete)
    payload, response = client.complete_json("judge", "evidence", {"verdict": "string"})
    assert payload == {"ok": True}
    assert '"verdict": "string"' in captured["system"]
    assert (response.prompt_tokens, response.completion_tokens) == (12, 4)
    assert response.cost_usd > 0

    unpriced_client = LLMClient(ProviderConfig("openai", "http://example", "gpt-4o-mini"))
    monkeypatch.setattr(unpriced_client._client, "complete_with_usage", complete)
    _payload, unpriced_response = unpriced_client.complete_json("judge", "evidence", {"verdict": "string"})
    assert unpriced_response.cost_usd is None
    assert unpriced_response.unpriced_model is True


def test_parse_reports_embedded_image_count():
    # The real fixture is known to contain no images; the field must still be explicit.
    doc = parse_pdf(Path(__file__).parents[1] / "Sample Contract.pdf")
    assert doc.embedded_image_count == 0
