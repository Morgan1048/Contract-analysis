from pathlib import Path
from unittest.mock import Mock

import pytest

from core.ingest.parse import parse_pdf
from core.retrieval.embeddings import FakeEmbedder
from core.retrieval.index import ChunkIndex
from core.retrieval.vectorstore import VectorStore
from core.rubric.schema import Finding, QuestionResult
from core.store import RunStore
from core.pipeline import AnalysisResult
from core.tools.contract_tools import build_tool_registry
from core.tools.evidence import build_evidence_registry
from core.tools.registry import Tool, ToolRegistry

FIXTURE = Path(__file__).parents[1] / "Sample Contract.pdf"


def _index():
    doc = parse_pdf(FIXTURE)
    return ChunkIndex(doc, store=VectorStore(FakeEmbedder(), persist_dir=".index/test-tools"))


def _registry(tmp_path, index=None):
    store = RunStore(tmp_path / "runs.db")
    return build_tool_registry(index or _index(), store), store


# 1. specs() and mcp_specs() both emit valid JSON Schema for all registry tools


def test_specs_and_mcp_specs_cover_all_five_tools(tmp_path):
    registry, _ = _registry(tmp_path)
    expected = {
        "search_evidence", "read_context", "read_table", "resolve_reference", "find_modifiers", "get_audit_trail",
        "search_contract",
        "get_section",
        "get_requirement",
        "list_deviations",
        "get_compliance_result",
        "explain_kpi",
        "get_kpi_snapshot",
        "get_kpi_history",
        "list_runs",
        "get_run_findings",
    }
    specs = registry.specs()
    assert {spec["function"]["name"] for spec in specs} == expected
    for spec in specs:
        assert spec["type"] == "function"
        assert isinstance(spec["function"]["parameters"], dict)
        assert spec["function"]["parameters"]["type"] == "object"

    mcp_specs = registry.mcp_specs()
    assert {spec["name"] for spec in mcp_specs} == expected
    for spec in mcp_specs:
        assert isinstance(spec["inputSchema"], dict)
        assert spec["inputSchema"]["type"] == "object"
        assert spec["description"]


# 2. call() with invalid arguments returns a structured error, raising nothing


def test_call_with_invalid_arguments_returns_structured_error(tmp_path):
    registry, _ = _registry(tmp_path)
    result = registry.call("get_section", {})
    assert "error" in result
    assert result["chunk_id"] is None

    result = registry.call("search_contract", {"query": 5})
    assert "error" in result


@pytest.mark.parametrize(("radius", "bound"), [(-1, ">= 0"), (13, "<= 12")])
def test_read_context_rejects_out_of_bounds_radius_before_handler(radius, bound):
    index = Mock()
    registry = build_evidence_registry(index)
    result = registry.call("read_context", {"chunk_id": "source", "radius": radius})
    index.context_window.assert_not_called()
    assert result["found"] is False
    assert result["chunk_id"] is None
    assert "radius" in result["error"] and bound in result["error"]
    assert "ValueError" not in result["error"]


@pytest.mark.parametrize("radius", [0, 12])
def test_read_context_accepts_inclusive_radius_bounds(radius):
    index = Mock()
    index.context_window.return_value = []
    index.context.return_value = []
    result = build_evidence_registry(index).call("read_context", {"chunk_id": "source", "radius": radius})
    assert "error" not in result
    index.context_window.assert_called_once_with("source", radius)


@pytest.mark.parametrize(("value", "valid"), [
    (0.4, False), (0.5, True), (1.5, True), (1.6, False),
    ("0.5", False), (True, False), (None, False),
])
def test_numeric_schema_bounds_are_validated_before_handler(value, valid):
    handler = Mock(return_value={"found": True})
    registry = ToolRegistry()
    registry.register(Tool("bounded", "Bounded number", {
        "type": "object", "properties": {"value": {"type": "number", "minimum": 0.5, "maximum": 1.5}},
    }, handler))
    result = registry.call("bounded", {"value": value})
    if valid:
        assert result == {"found": True}
        handler.assert_called_once_with(value=value)
    else:
        assert result["error"]
        assert result["found"] is False
        handler.assert_not_called()


# 3. call() with an unknown tool name returns a structured error


def test_call_with_unknown_tool_returns_structured_error(tmp_path):
    registry, _ = _registry(tmp_path)
    result = registry.call("delete_everything", {})
    assert "error" in result
    assert result["chunk_id"] is None


# 4. search_contract returns chunk_ids that exist in the parsed document


def test_search_contract_returns_real_chunk_ids(tmp_path):
    index = _index()
    registry, _ = _registry(tmp_path, index)
    real_ids = {chunk.chunk_id for chunk in index.doc.chunks}
    result = registry.call("search_contract", {"query": "password rotation break-glass", "k": 5})
    assert result["found"] is True
    assert result["results"]
    for item in result["results"]:
        assert item["chunk_id"] in real_ids


# 5. get_requirement("GOV-04") returns the exhibit row


def test_get_requirement_returns_exhibit_row(tmp_path):
    registry, _ = _registry(tmp_path)
    result = registry.call("get_requirement", {"requirement_id": "GOV-04"})
    assert result["found"] is True
    assert any(item["requirement_id"] == "GOV-04" for item in result["results"])


# 6. list_deviations returns RA-2026-001 with status "Open"


def test_list_deviations_returns_open_risk_acceptance(tmp_path):
    registry, _ = _registry(tmp_path)
    result = registry.call("list_deviations", {})
    assert result["found"] is True
    by_id = {item["risk_id"]: item for item in result["deviations"]}
    assert by_id["RA-2026-001"]["status"] == "Open"
    assert by_id["RA-2026-001"]["chunk_id"] == "risk:H:RA-2026-001"


# 7. get_compliance_result reads a stored run back from the run store


def test_get_compliance_result_reads_stored_run(tmp_path):
    registry, store = _registry(tmp_path)
    finding = Finding("P1", "MET", 0.9, "quote text", "sec:6.6", "reason")
    result = AnalysisResult(
        "run-1", "contract.pdf",
        [QuestionResult("Q1", "Question", "Fully Compliant", 0.9, [finding], [], "reason")],
        {}, {"prompt_tokens": 1, "completion_tokens": 1, "cost_usd": 0.0},
        "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:01+00:00",
    )
    store.save_run(result)
    fetched = registry.call("get_compliance_result", {"run_id": "run-1"})
    from core.output import to_output_document
    assert fetched == to_output_document(result, None)
    assert fetched["status"] == "completed"
    assert fetched["check_results"][0]["source_chunk_id"] == "sec:6.6"

    scoped = registry.call("get_compliance_result", {"run_id": "run-1", "question_id": "Q1"})
    assert len(scoped["questions"]) == 1

    missing = registry.call("get_compliance_result", {"run_id": "does-not-exist"})
    assert missing["found"] is False
    assert missing["chunk_id"] is None


# 8. every tool returns chunk_id or an explicit not-found marker — never an empty dict


def test_every_tool_returns_chunk_id_or_not_found_marker(tmp_path):
    registry, _ = _registry(tmp_path)
    not_found_section = registry.call("get_section", {"section_id": "99.99"})
    assert not_found_section != {}
    assert not_found_section["found"] is False
    assert "chunk_id" in not_found_section

    not_found_requirement = registry.call("get_requirement", {"requirement_id": "ZZZZ-99"})
    assert not_found_requirement != {}
    assert not_found_requirement["found"] is False

    empty_query = registry.call("search_contract", {"query": "zzzzzzz_no_match_qqqqqqq"})
    assert empty_query != {}
    if empty_query["found"]:
        assert all("chunk_id" in item for item in empty_query["results"])
    else:
        assert "chunk_id" in empty_query


def test_explain_kpi_returns_real_definition_and_formula_for_every_metric(tmp_path):
    registry = build_tool_registry(None, RunStore(tmp_path / "runs.db"))
    valid_names = registry.call("explain_kpi", {"name": "not_a_metric"})["valid_names"]
    placeholder_definition = "Operational KPI for the contract analyzer."
    placeholder_formula = "Derived from the current operational snapshot and evaluated using the same threshold logic as the dashboard."

    for name in valid_names:
        result = registry.call("explain_kpi", {"name": name})
        assert result["found"] is True
        assert result["definition"].strip()
        assert result["formula"].strip()
        assert result["definition"] != placeholder_definition
        assert result["formula"] != placeholder_formula


def test_source_tools_explain_reupload_when_historical_index_is_absent(tmp_path):
    registry = build_tool_registry(None, RunStore(tmp_path / "runs.db"))

    calls = [
        ("search_contract", {"query": "passwords"}),
        ("get_section", {"section_id": "6.6"}),
        ("get_requirement", {"requirement_id": "PASS-01"}),
        ("list_deviations", {}),
    ]
    for name, arguments in calls:
        result = registry.call(name, arguments)
        assert result["found"] is False
        assert result["source_required"] is True
        assert "original PDF" in result["message"]
        assert "Re-upload" in result["message"]
