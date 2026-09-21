from pathlib import Path

from core.store import RunStore
from core.tools.contract_tools import build_mcp_registry
from mcp_server import dispatch


def test_mcp_specs_are_registry_owned(tmp_path):
    registry = build_mcp_registry(store=RunStore(tmp_path / "runs.db"))
    names = {item["name"] for item in registry.mcp_specs()}
    assert names == {
        "search_evidence", "read_context", "read_table", "resolve_reference", "find_modifiers", "get_audit_trail",
        "analyze_contract",
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
    for item in registry.mcp_specs():
        assert item["name"] and item["description"]
        assert item["inputSchema"]["type"] == "object"


def test_unknown_mcp_tool_is_structured(tmp_path):
    registry = build_mcp_registry(store=RunStore(tmp_path / "runs.db"))
    response = dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "missing", "arguments": {}}},
        registry,
    )
    assert response["result"]["isError"] is True
    assert "error" in response["result"]["structuredContent"]


def test_analyze_returns_without_waiting(tmp_path):
    registry = build_mcp_registry(store=RunStore(tmp_path / "runs.db"))
    result = registry.call("analyze_contract", {"pdf_path": str(Path("does-not-exist.pdf"))})
    assert result["run_id"]
    assert result["status"] == "running"


def test_running_job_exposes_wait_and_clears_it_on_resume(tmp_path, monkeypatch):
    import core.ingest.parse as parsing
    import core.tools.contract_tools as tools
    from core.providers.progress import WaitUpdate, observer
    from tests.test_app_historical_runs import _stored_result

    store = RunStore(tmp_path / "runs.db")
    jobs = tools._MCPJobs(store, object())
    jobs.jobs["pending"] = {"status": "running"}
    path = tmp_path / "fixture.pdf"
    path.write_bytes(b"fixture")
    result = _stored_result(str(path))
    poll = tools._make_get_compliance_result(store, jobs=jobs)

    def analyze(*_args, **_kwargs):
        from dataclasses import replace
        assert _kwargs["run_id"] == "pending"
        observer()(WaitUpdate("example", "model", "rate_limit", 600, 1, 12, 429))
        status = poll("pending")
        assert status["status"] == "running"
        assert status["wait"]["remaining_s"] == 600
        observer()(WaitUpdate("example", "model", "rate_limit", 0, 1, 12, 429, "resumed"))
        assert "wait" not in poll("pending")
        return replace(result, run_id=_kwargs["run_id"])

    monkeypatch.setattr(parsing, "parse_pdf", lambda *_: object())
    monkeypatch.setattr(tools, "ChunkIndex", lambda *_a, **_k: object())
    monkeypatch.setattr(tools, "analyze", analyze)
    jobs._run("pending", str(path), None)
    assert poll("pending")["status"] == "completed"
    assert "wait" not in poll("pending")
