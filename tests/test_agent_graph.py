from dataclasses import replace
import json
from pathlib import Path

import pytest
from jsonschema import validate

from core.agents.evaluator import verify
from core.agents.graph import Limits
from core.ingest.chunk import chunk_blocks
from core.ingest.parse import parse_pdf
from core.kpi import snapshot
from core.llm.stub import StubLLM
from core.models import Chunk, ParsedDocument, SourceBlock
from core.openapi_schema import analysis_result_schema
from core.output import to_output_document
from core.pipeline import analyze
from core.retrieval.embeddings import FakeEmbedder
from core.retrieval.index import ChunkIndex
from core.retrieval.router import EvidencePack
from core.retrieval.vectorstore import VectorStore
from core.rubric.aggregate import load_rubric
from core.rubric.schema import Finding
from core.store import RunStore
from eval.synthetic import rubric, write_contract


def make_index(tmp_path, *, renumber=False):
    path = write_contract(tmp_path / "synthetic.pdf", renumber=renumber)
    doc = parse_pdf(path)
    return path, ChunkIndex(doc, VectorStore(FakeEmbedder(), tmp_path / "index"))


def test_rubric_is_document_independent():
    questions = load_rubric()
    checks = [s for q in questions.values() for s in q.sub_requirements]
    assert len(checks) == 25
    assert all(s.facets and not s.requirement_ids and not s.section_ids for s in checks)
    assert "not mandatory" in questions["Q4"].sub_requirements[0].guidance
    assert "not required" in questions["Q5"].sub_requirements[0].guidance


def test_table_without_control_ids_preserves_headers_and_cells():
    blocks = [SourceBlock("Training Once per year", 2, "table_row",
                          (1, 2, 3, 4), "table:1", ("Training", "Once per year"), ("Control", "Frequency"))]
    chunks = chunk_blocks(blocks)
    assert len(chunks) == 1
    assert chunks[0].kind == "table_row"
    assert chunks[0].requirement_id is None
    assert chunks[0].cells == ("Training", "Once per year")
    assert "Frequency" in chunks[0].retrieval_text
    assert chunks[0].text == blocks[0].text


def test_prose_without_ids_and_renumbered_sections_are_searchable(tmp_path):
    _, index = make_index(tmp_path, renumber=True)
    assert index.search("recyclable outer cartons")
    assert any("recyclable" in c.text for c, _ in index.search("recyclable outer cartons"))
    assert not any(c.requirement_id for c in index.doc.chunks)


def test_quotation_verification_does_not_erase_word_boundaries():
    chunk = Chunk("source", "clause", "Supplier mustnot share secrets.", 1, None, None, None, None)
    finding = Finding("x", "MET", .9, "must not", "source", "reason")
    checks = verify(finding, EvidencePack("x", [chunk], []), ParsedDocument("x", 1, chunk.text, [chunk]))
    assert not next(c for c in checks if c.name == "quote_verbatim").passed


def test_not_found_without_fabricated_quote_passes_mechanical_verification():
    finding = Finding("x", "NOT_FOUND", .8, "", "", "No matching obligation.")
    assert all(c.passed for c in verify(finding, EvidencePack("x", [], []), ParsedDocument("x", 1, "", [])))


def test_graph_and_audit_round_trip_all_25_checks(tmp_path):
    path, index = make_index(tmp_path)
    result = analyze(path, StubLLM(), rubric=rubric(), index=index, assessment_date="2026-01-15")
    assert not result.incomplete
    assert sum(len(q.findings) for q in result.questions) == 25
    assert {e["node"] for e in result.audit_events} == {"router", "retrieve", "judge", "verify", "critic"}
    assert result.usage["quote_attempts"] == 25
    assert all(f.search_complete for q in result.questions for f in q.findings)
    store = RunStore(tmp_path / "runs.db")
    store.save_run(result)
    saved = store.get_run(result.run_id)
    assert saved == result
    assert len(store.audit_trail(result.run_id)) == 25
    output = to_output_document(saved, index)
    validate(output, analysis_result_schema())
    assert len(output["questions"]) == 5 and not output["review_required"]
    assert snapshot(store)["completion_rate"] == 1
    assert snapshot(store)["instrumented_runs"] == 1
    assert snapshot(store)["followup_resolution_rate"] is None
    from core.monitoring import usage_snapshot
    usage = usage_snapshot(store)
    assert sum(q["prompt_tokens"] for q in usage["question_cost"]) == result.usage["prompt_tokens"]
    assert sum(q["completion_tokens"] for q in usage["question_cost"]) == result.usage["completion_tokens"]


def test_budget_exhaustion_is_review_not_non_compliance(tmp_path):
    path, index = make_index(tmp_path)
    result = analyze(path, StubLLM(), rubric=rubric(), index=index, limits=Limits(llm_calls=0))
    output = to_output_document(result, index)
    assert result.incomplete and result.usage["budget_exhausted"]
    assert not output["questions"] and len(output["review_required"]) == 5
    validate(output, analysis_result_schema())


class FollowupClient(StubLLM):
    def __post_init__(self):
        super().__post_init__()
        self.critiques = 0

    def complete_json(self, system, user, schema, example=None):
        payload, response = super().complete_json(system, user, schema, example)
        if "ROLE: evidence_critic" in system:
            data = json.loads(user)
            assert set(data["search_history"]) == set(data["pending_check_ids"])
            assert all(v["modifier_scan_complete"] for v in data["source_coverage"].values())
            self.critiques += 1
            if self.critiques == 1:
                for item in payload["reviews"]:
                    item.update(decision="retrieve", gap="Check the dispatch timing in the delivery terms.")
        return payload, response


def test_critic_feedback_causes_targeted_second_retrieval(tmp_path):
    path, index = make_index(tmp_path)
    result = analyze(path, FollowupClient(), rubric={"Q1": rubric()["Q1"]}, index=index)
    assert not result.incomplete
    assert result.usage["followup_checks"] == 5
    assert result.usage["resolved_followup_checks"] == 5
    events = [e for e in result.audit_events if e["node"] == "router"]
    assert len(events) == 2 and events[1]["round"] == 1
    assert "dispatch timing" in events[1]["actions"][0]["arguments"]["query"]


def test_one_unresolved_check_keeps_all_five_questions_in_output(tmp_path):
    class UnresolvedCheckClient(StubLLM):
        def complete_json(self, system, user, schema, example=None):
            payload, response = super().complete_json(system, user, schema, example)
            if "ROLE: evidence_critic" in system:
                for review in payload["reviews"]:
                    if review["check_id"] == "weekly_dispatch":
                        review.update(decision="unresolved", gap="Required delivery attachment is unavailable.")
            return payload, response

    path, index = make_index(tmp_path)
    result = analyze(path, UnresolvedCheckClient(), rubric=rubric(), index=index)
    assert result.questions[0].state == "Partially Compliant"
    assert result.incomplete
    assert result.questions[0].confidence == 0.0
    store = RunStore(tmp_path / "unresolved.db")
    store.save_run(result)
    saved = store.get_run(result.run_id)
    assert saved == result
    output = to_output_document(saved, index)
    validate(output, analysis_result_schema())
    assert output["status"] == "incomplete"
    assert len(output["questions"]) == 5
    assert output["review_required"] == []
    assert "weekly_dispatch: UNRESOLVED" in output["questions"][0]["Rationale"]
    assert "Required delivery attachment is unavailable." in output["questions"][0]["Rationale"]
    assert output["questions"][0]["Confidence"] == 0.0
    assert len(output["check_results"]) == 25
    assert next(f for f in output["check_results"]
                if f["sub_requirement_id"] == "weekly_dispatch")["verdict"] == "UNRESOLVED"


@pytest.mark.parametrize("verdict", ["MET", "PARTIAL", "CONTRADICTED", "NOT_FOUND"])
def test_verified_finding_survives_exhausted_followup_budget(tmp_path, verdict):
    class VerdictFollowupClient(FollowupClient):
        def complete_json(self, system, user, schema, example=None):
            payload, response = super().complete_json(system, user, schema, example)
            if "ROLE: compliance_judge" in system:
                payload.update(verdict=verdict, confidence=0.73,
                               rationale="Verified round-zero assessment.",
                               gaps=["Original evidence gap."])
                for facet in payload["facet_results"]:
                    facet["verdict"] = verdict
                    if verdict == "NOT_FOUND":
                        facet["evidence_indices"] = []
                if verdict == "NOT_FOUND":
                    payload.update(quote="", source_chunk_id="", evidence=[])
            return payload, response

    path, index = make_index(tmp_path)
    result = analyze(path, VerdictFollowupClient(), rubric={"Q1": rubric()["Q1"]},
                     index=index, limits=Limits(followups=0))
    assert result.audit_events[-1]["stop_reason"] == "followup_budget"
    assert result.audit_events[-1]["round"] == 0
    for finding in result.questions[0].findings:
        assert finding.verdict == verdict
        assert finding.confidence == 0.73
        assert finding.quote_verified
        assert not finding.search_complete
        assert finding.rationale == "Verified round-zero assessment."
        reason = result.audit_events[-1]["feedback"][finding.sub_requirement_id]
        assert finding.gaps == ["Original evidence gap.", reason]
        assert any(c.name == "analysis_complete" and not c.passed and c.detail == reason
                   for c in result.checks[finding.sub_requirement_id])
    assert result.incomplete and result.usage["budget_exhausted"]
    output = to_output_document(result, index)
    assert output["status"] == "incomplete"
    assert len(output["questions"]) == 1
    assert output["review_required"] == []


def test_unverified_finding_stays_unresolved_on_followup_exhaustion(tmp_path):
    class InvalidQuoteClient(StubLLM):
        def complete_json(self, system, user, schema, example=None):
            payload, response = super().complete_json(system, user, schema, example)
            if "ROLE: compliance_judge" in system:
                payload["evidence"][0]["quote"] = "Fabricated evidence not present in the document."
            return payload, response

    path, index = make_index(tmp_path)
    result = analyze(path, InvalidQuoteClient(), rubric={"Q1": rubric()["Q1"]},
                     index=index, limits=Limits(followups=0))
    assert result.audit_events[-1]["stop_reason"] == "followup_budget"
    assert result.incomplete
    assert result.questions[0].state == "Indeterminate"
    for finding in result.questions[0].findings:
        assert finding.verdict == "UNRESOLVED"
        assert finding.confidence == 0
        assert not finding.quote_verified
        assert not finding.search_complete
        assert any(c.name == "analysis_complete" and not c.passed
                   for c in result.checks[finding.sub_requirement_id])


def test_critic_schema_repair_is_bounded_and_audited(tmp_path):
    class RepairClient(StubLLM):
        critic_calls = 0

        def complete_json(self, system, user, schema, example=None):
            payload, response = super().complete_json(system, user, schema, example)
            if "ROLE: evidence_critic" in system:
                self.critic_calls += 1
                if self.critic_calls == 1:
                    payload["reviews"][0]["check_id"] = "wrong-id"
                else:
                    assert "wrong-id" in json.loads(user)["output_repair"]
            return payload, response

    path, index = make_index(tmp_path)
    result = analyze(path, RepairClient(), rubric={"Q1": rubric()["Q1"]}, index=index)
    assert not result.incomplete
    event = next(e for e in result.audit_events if e["node"] == "critic")
    assert event["llm_calls"] == 2
    assert event["schema_repairs"][0]["agent"] == "critic"
    assert len(event["reviews"]) == 5


def test_large_critic_batch_splits_without_omitting_checks(tmp_path):
    from core.llm.client import PromptBudgetExceeded

    class BoundedCritic(StubLLM):
        batches = []

        def validate_prompt(self, system, user, schema, example=None):
            if "ROLE: evidence_critic" in system:
                ids = json.loads(user)["pending_check_ids"]
                if len(ids) > 1:
                    raise PromptBudgetExceeded("prompt_token_budget")
                self.batches.append(ids)

    path, index = make_index(tmp_path)
    client = BoundedCritic()
    result = analyze(path, client, rubric={"Q1": rubric()["Q1"]}, index=index)
    assert not result.incomplete
    assert len(client.batches) == 5
    assert result.usage["llm_calls"] == 11
    reviews = next(e for e in result.audit_events if e["node"] == "critic")["reviews"]
    assert {r["check_id"] for r in reviews} == {s.id for s in rubric()["Q1"].sub_requirements}


def test_context_window_preserves_header_without_expanding_entire_table(tmp_path):
    from core.tools.evidence import build_evidence_registry
    blocks = [SourceBlock(f"Row {i} provision", 1, "table_row", table_id="table",
                          cells=(str(i), "provision"), headers=("Term", "Value")) for i in range(100)]
    chunks = chunk_blocks(blocks)
    doc = ParsedDocument("fixture", 1, " ".join(c.text for c in chunks), chunks)
    index = ChunkIndex(doc, VectorStore(FakeEmbedder(), tmp_path / "bounded"))
    registry = build_evidence_registry(index)
    selected = registry.call("read_context", {"chunk_id": chunks[50].chunk_id})
    assert selected["available_sources"] == 100
    assert len(selected["results"]) == 6
    assert selected["results"][0]["chunk_id"] == chunks[0].chunk_id
    assert len(registry.call("read_table", {"chunk_id": chunks[50].chunk_id})["results"]) == 100


def test_unreadable_page_cannot_produce_completed_compliance(tmp_path):
    path, index = make_index(tmp_path)
    warning = "Page 1: insufficient readable text; local OCR/manual review required."
    index.doc = replace(index.doc, unreadable_pages=[1], warnings=[warning])
    result = analyze(path, StubLLM(), rubric={"Q1": rubric()["Q1"]}, index=index)
    assert result.incomplete
    assert warning in result.usage["parse_warnings"]
    assert result.questions[0].state in {"Fully Compliant", "Partially Compliant", "Non-Compliant"}
    assert all(f.quote_verified and not f.search_complete for f in result.questions[0].findings)
    assert all(any("unreadable" in gap for gap in f.gaps) for f in result.questions[0].findings)
    assert all(any(c.name == "analysis_complete" and not c.passed for c in checks)
               for checks in result.checks.values())
    output = to_output_document(result, index)
    validate(output, analysis_result_schema())
    assert output["status"] == "incomplete"
    assert len(output["questions"]) == 1
    assert output["review_required"] == []


def test_missing_and_fabricated_model_fields_do_not_become_non_compliance(tmp_path):
    path, index = make_index(tmp_path)
    result = analyze(path, StubLLM(mode="malformed"), rubric={"Q1": rubric()["Q1"]}, index=index)
    assert result.incomplete
    assert result.audit_events[0]["error"]


def test_explicit_remote_opt_in_required_before_any_inference(tmp_path):
    path, index = make_index(tmp_path)
    from core.llm.client import LLMClient
    from core.providers.config import ProviderConfig
    with pytest.raises(ValueError, match="explicit authorization"):
        analyze(path, LLMClient(ProviderConfig("openai", "https://api.openai.com/v1", "test")),
                rubric=rubric(), index=index)


def test_legacy_or_unapproved_chat_cannot_send_history_to_hosted_model(tmp_path):
    from core.agents.chat import ChatAgent
    from core.llm.client import LLMClient
    from core.providers.config import ProviderConfig
    from core.tools.registry import ToolRegistry
    store = RunStore(tmp_path / "chat.db")
    client = LLMClient(ProviderConfig("openai", "https://api.openai.com/v1", "test"))
    agent = ChatAgent(client, ToolRegistry(), store, "test-session", None)
    with pytest.raises(ValueError, match="Remote chat requires"):
        agent.ask("Do not transmit this question")
    assert agent.history() == []


def test_actual_pdf_table_without_ids_is_not_flattened_into_prose(tmp_path):
    from eval.variants import _write_text_pdf
    commands = ["0.5 w"]
    for x in (40, 200, 550):
        commands.append(f"{x} 600 m {x} 720 l S")
    for y in (600, 640, 680, 720):
        commands.append(f"40 {y} m 550 {y} l S")
    for x, y, text in [
        (50, 700, "Control"), (210, 700, "Requirement"),
        (50, 660, "Carton handling"), (210, 660, "Cartons must be sealed."),
        (50, 620, "Receipt handling"), (210, 620, "Carrier receipts must be archived."),
    ]:
        commands.append(f"BT /F1 10 Tf {x} {y} Td ({text}) Tj ET")
    path = tmp_path / "unrelated-table.pdf"
    _write_text_pdf(["Schedule Z - Supply terms"], path, drawing_commands=commands)
    doc = parse_pdf(path)
    row = next(c for c in doc.chunks if "Cartons must be sealed." in c.text)
    assert row.kind == "table_row"
    assert row.requirement_id is None
    assert row.headers == ("Control", "Requirement")
    assert row.cells == ("Carton handling", "Cartons must be sealed.")
    assert row.bbox and row.exhibit_id == "Z"
    assert row.text in doc.text


def test_explicit_sections_and_cross_page_clause_context():
    blocks = [
        SourceBlock("Section 42 Delivery terms. Shipments must use sealed cartons.", 1),
        SourceBlock("Each carton must carry an order label.", 2),
        SourceBlock("Article IX Returns. Returned goods must be logged.", 2),
    ]
    chunks = chunk_blocks(blocks)
    assert chunks[0].section_id == chunks[1].section_id == "42"
    assert chunks[0].parent_id == chunks[1].parent_id
    assert chunks[1].page == 2
    assert chunks[2].section_id == "IX"


def test_same_header_different_tables_do_not_share_parent_context(tmp_path):
    # Distinct table IDs remain distinct unless the parser identifies continuation.
    chunks = chunk_blocks([
        SourceBlock("Term One", 1, "table_row", table_id="a", cells=("Term", "One"), headers=("Name", "Value")),
        SourceBlock("Term Two", 1, "table_row", table_id="b", cells=("Term", "Two"), headers=("Name", "Value")),
    ])
    doc = ParsedDocument("fixture", 1, "Term One Term Two", chunks)
    index = ChunkIndex(doc, VectorStore(FakeEmbedder(), tmp_path / "tables"))
    assert index.context(chunks[0].chunk_id) == [chunks[0]]


def test_parser_links_cross_page_continuation_not_same_page_tables(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import core.ingest.parse as parser
    pages = [SimpleNamespace(images=[], chars=["x"] * 50) for _ in range(2)]
    headers = ("Name", "Value")
    blocks = {
        1: [SourceBlock("Alpha One", 1, "table_row", table_id="a", headers=headers),
            SourceBlock("Beta Two", 1, "table_row", table_id="b", headers=headers)],
        2: [SourceBlock("Gamma Three", 2, "table_row", table_id="c", headers=headers),
            SourceBlock("Delta Four", 2, "table_row", table_id="c", headers=headers)],
    }
    monkeypatch.setattr(parser.pdfplumber, "open", lambda _: nullcontext(SimpleNamespace(pages=pages)))
    monkeypatch.setattr(parser, "_page_blocks", lambda _, number: (blocks[number], []))
    chunks = parser.parse_pdf("fixture.pdf").chunks
    assert chunks[0].table_id != chunks[1].table_id
    assert chunks[1].table_id == chunks[2].table_id == chunks[3].table_id


def test_ambiguous_columns_are_flagged_instead_of_silently_interleaved(tmp_path):
    from eval.variants import _write_text_pdf
    commands = []
    for y in (700, 680, 660):
        for x, text in ((50, "Left prose"), (360, "Right prose")):
            commands.append(f"BT /F1 10 Tf {x} {y} Td ({text}) Tj ET")
    path = tmp_path / "columns.pdf"
    _write_text_pdf(["Independent layout"], path, drawing_commands=commands)
    doc = parse_pdf(path)
    assert doc.unreadable_pages == [1]
    assert any("reading order" in warning for warning in doc.warnings)


def test_streamlit_renders_persisted_graph_result(tmp_path, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    path, index = make_index(tmp_path)
    result = analyze(path, StubLLM(), rubric=rubric(), index=index)
    db_path = tmp_path / "ui.db"
    RunStore(db_path).save_run(result)
    monkeypatch.setenv("CA_RUN_DB", str(db_path))
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
        assert not app.exception
        app.sidebar.selectbox[0].select(result.run_id).run()
        assert not app.exception
        assert app.session_state["result"].run_id == result.run_id
    finally:
        st.cache_resource.clear()
