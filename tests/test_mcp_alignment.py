from dataclasses import asdict, replace
import io
import json
import threading
import time

import pytest
from jsonschema import validate

from core.agents.chat import _harvest_evidence
from core.llm.stub import StubLLM
from core.models import Chunk, ParsedDocument
from core.output import to_output_document
from core.pipeline import analyze
from core.providers.progress import WaitUpdate, observer
from core.retrieval.embeddings import FakeEmbedder
from core.retrieval.index import ChunkIndex
from core.retrieval.vectorstore import VectorStore
from core.store import RunStore
from core.tools import contract_tools as tools
from eval.synthetic import rubric, write_contract
from mcp_server import dispatch, serve_stdio
from tests.test_openapi import _openapi_document, _resolve_local_refs
from tests.test_store import _result


SOURCE_CALLS = [
    ("search_evidence", {"query": "exception"}),
    ("read_context", {"chunk_id": "shared-id"}),
    ("read_table", {"chunk_id": "shared-id"}),
    ("resolve_reference", {"chunk_id": "shared-id"}),
    ("find_modifiers", {}),
    ("search_contract", {"query": "exception"}),
    ("get_section", {"section_id": "1"}),
    ("get_requirement", {"requirement_id": "CTRL-01"}),
    ("list_deviations", {}),
]


def assert_poll_schema(result):
    document = _openapi_document()
    schema = _resolve_local_refs(document["components"]["schemas"]["RunResponse"], document)
    validate(result, schema)


def poll_until_done(registry, run_id):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        response = registry.call("get_compliance_result", {"run_id": run_id})
        if response.get("status") != "running":
            return response
        time.sleep(.01)
    pytest.fail("MCP job did not finish")


def test_async_full_graph_uses_one_id_and_survives_registry_restart(tmp_path, monkeypatch):
    path = write_contract(tmp_path / "independent.pdf")
    indexes = []

    def index(doc, **_kwargs):
        value = ChunkIndex(doc, VectorStore(FakeEmbedder(), tmp_path / "index"))
        indexes.append(value)
        return value

    def synthetic_analyze(*args, **kwargs):
        return analyze(*args, rubric=rubric(), **kwargs)

    monkeypatch.setattr(tools, "ChunkIndex", index)
    monkeypatch.setattr(tools, "analyze", synthetic_analyze)
    store = RunStore(tmp_path / "runs.db")
    registry = tools.build_mcp_registry(store, StubLLM())
    submitted = registry.call("analyze_contract", {"pdf_path": str(path)})
    assert submitted["status"] == "running"
    run_id = submitted["run_id"]
    response = poll_until_done(registry, run_id)
    assert response["status"] == "completed", response
    saved = store.get_run(run_id)
    assert saved.run_id == run_id
    assert response == to_output_document(saved, indexes[0])
    assert len(response["check_results"]) == 25
    assert len(response["questions"]) == 5
    assert_poll_schema(response)
    assert registry.call("list_runs", {})["runs"][0]["run_id"] == run_id
    assert registry.call("get_run_findings", {"run_id": run_id})["found"]
    audit = registry.call("get_audit_trail", {"run_id": run_id})
    assert audit["found"] and audit["events"]
    assert all(event["trace_id"] == saved.usage["trace_id"] for event in audit["events"])
    assert [row[0] for row in store.db.execute("SELECT DISTINCT run_id FROM audit_events")] == [run_id]

    restarted = tools.build_mcp_registry(RunStore(store.path), StubLLM())
    assert restarted.call("get_compliance_result", {"run_id": run_id}) == response
    assert restarted.call("get_audit_trail", {"run_id": run_id}) == audit
    unavailable = restarted.call("search_evidence", {"run_id": run_id, "query": "cartons"})
    assert unavailable["code"] == "SOURCE_UNAVAILABLE" and unavailable["source_required"]
    assert restarted.call("get_compliance_result", {"run_id": run_id, "question_id": "absent"})["code"] == "QUESTION_NOT_FOUND"


def test_every_source_tool_requires_run_and_never_uses_latest_index(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "runs.db")
    jobs = tools._MCPJobs(store, StubLLM())
    for name in ("ALPHA", "BETA"):
        chunks = [
            Chunk("shared-id", "table_row", f"Exception {name}: see Section 2.", 1, "1",
                  None, "CTRL-01", "Matrix", table_id="table:1", cells=(name,), headers=("Control",)),
            Chunk("target-id", "clause", f"Section 2. {name} obligations.", 1, "2", None, None, "Terms"),
        ]
        doc = ParsedDocument(name, 1, "\n".join(c.text for c in chunks), chunks)
        jobs.indexes[name] = ChunkIndex(doc, VectorStore(FakeEmbedder(), tmp_path / name))
        store.save_run(_result(name))
    monkeypatch.setattr(tools, "_MCPJobs", lambda *_args: jobs)
    registry = tools.build_mcp_registry(store, StubLLM())
    for tool_name, arguments in SOURCE_CALLS:
        assert "run_id" in registry.get(tool_name).parameters["required"]
        assert "error" in registry.call(tool_name, arguments)
        unknown = registry.call(tool_name, {"run_id": "absent", **arguments})
        assert unknown["code"] == "RUN_NOT_FOUND"
        for run_id, other in (("ALPHA", "BETA"), ("BETA", "ALPHA")):
            result = registry.call(tool_name, {"run_id": run_id, **arguments})
            assert result["found"], (tool_name, result)
            assert result["run_id"] == run_id
            assert run_id in json.dumps(result)
            assert other not in json.dumps(result)


def test_waiting_and_failed_jobs_have_typed_poll_responses(tmp_path, monkeypatch):
    import core.ingest.parse as parsing
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(parsing, "parse_pdf", lambda *_args: object())
    monkeypatch.setattr(tools, "ChunkIndex", lambda *_a, **_kw: object())
    path = tmp_path / "fixture.pdf"
    path.write_bytes(b"fixture")

    def failing_analyze(*_args, **_kwargs):
        observer()(WaitUpdate("local", "test", "rate_limit", 600, 1, 12, 429))
        entered.set()
        assert release.wait(10)
        observer()(WaitUpdate("local", "test", "rate_limit", 0, 1, 12, 429, "resumed"))
        raise ValueError("controlled failure")

    monkeypatch.setattr(tools, "analyze", failing_analyze)
    registry = tools.build_mcp_registry(RunStore(tmp_path / "runs.db"), StubLLM())
    run_id = registry.call("analyze_contract", {"pdf_path": str(path)})["run_id"]
    try:
        assert entered.wait(5)
        waiting = registry.call("get_compliance_result", {"run_id": run_id})
        assert waiting["wait"]["remaining_s"] == 600
        assert_poll_schema(waiting)
        not_ready = registry.call("read_context", {"run_id": run_id, "chunk_id": "x"})
        assert not_ready["code"] == "RUN_NOT_READY"
    finally:
        release.set()
    failed = poll_until_done(registry, run_id)
    assert failed["status"] == "failed" and failed["code"] == "ANALYSIS_FAILED"
    assert "wait" not in failed
    assert_poll_schema(failed)
    assert registry.call("find_modifiers", {"run_id": run_id})["code"] == "ANALYSIS_FAILED"


@pytest.mark.parametrize("arguments", [{}, {"pdf_path": "x", "pdf_bytes": "eA=="}, {"pdf_path": ""}])
def test_invalid_intake_creates_no_job(tmp_path, monkeypatch, arguments):
    monkeypatch.setattr(threading.Thread, "start", lambda *_: pytest.fail("Started an invalid job"))
    registry = tools.build_mcp_registry(RunStore(tmp_path / "runs.db"), StubLLM())
    result = registry.call("analyze_contract", arguments)
    assert result["code"] == "INVALID_ARGUMENTS" and "run_id" not in result


def test_incomplete_result_uses_public_schema_and_only_verified_citations(tmp_path):
    result = _result("incomplete")
    finding = replace(result.questions[0].findings[0], quote_verified=True)
    resolved = replace(result.questions[0], findings=[finding])
    unresolved = replace(resolved, question_id="Q2", state="Indeterminate", rationale="Transport timeout",
                         findings=[replace(finding, sub_requirement_id="A1", verdict="UNRESOLVED",
                                           quote_verified=False, quote="", source_chunk_id="")])
    result = replace(result, questions=[resolved, unresolved], incomplete=True)
    store = RunStore(tmp_path / "runs.db")
    store.save_run(result)
    registry = tools.build_mcp_registry(store, StubLLM())
    response = registry.call("get_compliance_result", {"run_id": result.run_id})
    assert_poll_schema(response)
    assert response == to_output_document(result, None)
    assert len(response["questions"]) == 1
    assert response["review_required"][0]["question_id"] == "Q2"
    selected = registry.call("get_compliance_result", {"run_id": result.run_id, "question_id": "Q2"})
    assert selected["status"] == "incomplete" and selected["questions"] == []
    assert selected["review_required"] == response["review_required"]
    assert _harvest_evidence(response) == {"sec:6.6": "quote"}
    unverified = {"check_results": [asdict(replace(finding, quote_verified=False))]}
    assert _harvest_evidence(unverified) == {}


def test_snapshot_preserves_table_citation_without_live_source(tmp_path):
    chunk = Chunk("opaque-source", "table_row", "Supplier shall train personnel.", 2,
                  None, "Z", "TRAIN-01", "Z4. Personnel")
    result = _result("snapshot")
    finding = replace(result.questions[0].findings[0], quote=chunk.text,
                      source_chunk_id=chunk.chunk_id, quote_verified=True)
    result = replace(result, questions=[replace(result.questions[0], findings=[finding])],
                     evidence_snapshot={chunk.chunk_id: asdict(chunk)})
    store = RunStore(tmp_path / "runs.db")
    store.save_run(result)
    response = tools.build_mcp_registry(store, StubLLM()).call("get_compliance_result", {"run_id": result.run_id})
    assert response["questions"][0]["Relevant Quotes"] == [
        {"quote": chunk.text, "source_chunk_id": chunk.chunk_id, "citation": "Exhibit Z4 (TRAIN-01)"}
    ]


def test_get_compliance_result_emits_assignment_fields(tmp_path):
    result = _result("public-fields")
    store = RunStore(tmp_path / "runs.db")
    store.save_run(result)
    registry = tools.build_mcp_registry(store, StubLLM())
    response = registry.call("get_compliance_result", {"run_id": result.run_id})
    assert response == to_output_document(result, None)
    assert response["run_id"] == result.run_id
    assert response["status"] == "completed"
    assert len(response["questions"]) == len(result.questions)
    for question in response["questions"]:
        assert set(question) == {
            "Compliance Question", "Compliance State", "Confidence", "Relevant Quotes", "Rationale",
        }
    assert_poll_schema(response)


def test_stdio_handshake_and_poll_preserve_public_payload(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    store.save_run(_result("stored"))
    registry = tools.build_mcp_registry(store, StubLLM())
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "get_compliance_result", "arguments": {"run_id": "stored"}}},
    ]
    stdout = io.StringIO()
    serve_stdio(registry, io.StringIO("\n".join(json.dumps(m) for m in messages)), stdout)
    handshake, response = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert handshake["result"]["serverInfo"]["name"] == "contract-analyzer"
    result = response["result"]
    assert not result["isError"]
    assert json.loads(result["content"][0]["text"]) == result["structuredContent"]
    assert_poll_schema(result["structuredContent"])
    missing = dispatch({"id": 3, "method": "tools/call", "params": {
        "name": "get_compliance_result", "arguments": {"run_id": "missing"}}}, registry)
    assert missing["result"]["isError"]


def test_executable_stdio_reads_same_configured_database(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path

    store = RunStore(tmp_path / "ui-runs.db")
    result = _result("ui-run")
    store.save_run(result)
    monkeypatch.setenv("CA_RUN_DB", str(store.path))
    message = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
        "name": "get_compliance_result", "arguments": {"run_id": result.run_id}}}
    process = subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "mcp_server.py")],
        cwd=tmp_path, input=json.dumps(message) + "\n", capture_output=True, text=True, timeout=20,
    )
    assert process.returncode == 0, process.stderr
    lines = process.stdout.splitlines()
    assert len(lines) == 1
    response = json.loads(lines[0])["result"]
    assert not response["isError"]
    assert response["structuredContent"] == to_output_document(result, None)
