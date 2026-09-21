"""Shared evidence, result and KPI tools with explicit run-scoped MCP adapters."""
from __future__ import annotations

import base64
import os
import tempfile
import threading
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from core.kpi import (
    evaluate_thresholds,
    history as kpi_history,
    kpi_explanation,
    kpi_metric_names,
    snapshot,
)
from core.llm.client import LLMClient
from core.pipeline import analyze
from core.output import to_output_document
from core.obs import get_logger
from core.obs.trace import span
from core.providers.config import ProviderConfig
from core.retrieval.deviations import extract_deviations
from core.retrieval.index import ChunkIndex
from core.tools.registry import Tool, ToolRegistry
from core.tools.evidence import build_evidence_registry


def _chunk_view(chunk: Any) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "page": chunk.page,
        "kind": chunk.kind,
        "section_id": chunk.section_id,
        "exhibit_id": chunk.exhibit_id,
        "requirement_id": chunk.requirement_id,
        "heading": chunk.heading,
    }


def _not_found(**extra: Any) -> dict[str, Any]:
    return {"chunk_id": None, "found": False, **extra}


def _source_unavailable(**extra: Any) -> dict[str, Any]:
    return _not_found(
        source_required=True,
        message=(
            "The original contract source is not available for this historical run. "
            "Re-upload the original PDF to perform new retrieval or rerun analysis."
        ),
        **extra,
    )


def _make_search_contract(index):
    def search_contract(query: str, k: int = 6, kinds: list[str] | None = None) -> dict:
        if index is None:
            return _source_unavailable(query=query, results=[])
        results = index.search(query, k=k, kinds=tuple(kinds) if kinds else None)
        if not results:
            return _not_found(query=query, results=[])
        return {"found": True, "results": [dict(_chunk_view(chunk), score=score) for chunk, score in results]}

    return search_contract


def _make_get_section(index):
    def get_section(section_id: str) -> dict:
        if index is None:
            return _source_unavailable(section_id=section_id)
        chunks = index.by_section_id(section_id)
        if not chunks:
            return _not_found(section_id=section_id)
        return {"found": True, "results": [_chunk_view(chunk) for chunk in chunks]}

    return get_section


def _make_get_requirement(index):
    def get_requirement(requirement_id: str) -> dict:
        if index is None:
            return _source_unavailable(requirement_id=requirement_id)
        chunks = index.by_requirement_id(requirement_id)
        if not chunks:
            return _not_found(requirement_id=requirement_id)
        return {"found": True, "results": [_chunk_view(chunk) for chunk in chunks]}

    return get_requirement


def _make_list_deviations(index):
    def list_deviations() -> dict:
        if index is None:
            return _source_unavailable(deviations=[])
        deviations = extract_deviations(index.doc)
        if not deviations:
            return _not_found(deviations=[])
        return {
            "found": True,
            "deviations": [
                {
                    "chunk_id": deviation.source_chunk_id,
                    "risk_id": deviation.risk_id,
                    "status": deviation.status,
                    "requirement_ids": deviation.requirement_ids,
                    "text": deviation.text,
                    "expiration": deviation.expiration,
                    "candidate_only": True,
                }
                for deviation in deviations
            ],
        }

    return list_deviations


def _kpi_metric_names() -> list[str]:
    return kpi_metric_names()


def _kpi_alert_for_name(name: str, snapshot_dict: dict, *, calibration_bins: list[dict] | None = None, accuracy: float | None = None, state_distribution_shift: float | None = None):
    alerts = evaluate_thresholds(
        snapshot_dict,
        calibration_bins=calibration_bins,
        accuracy=accuracy,
        state_distribution_shift=state_distribution_shift,
    )
    for alert in alerts:
        if alert.name == name:
            return alert
    return None


def _make_explain_kpi(store):
    def explain_kpi(name: str) -> dict:
        valid_names = _kpi_metric_names()
        if name not in valid_names:
            return {"error": f"unknown KPI: {name!r}", "valid_names": valid_names, "found": False}

        snap = snapshot(store)
        alert = _kpi_alert_for_name(name, snap)
        explanation = kpi_explanation(name, snap)
        if alert is not None:
            explanation["threshold"] = alert.threshold
            explanation["severity"] = alert.severity
            explanation["current_value"] = alert.value
            explanation["action"] = alert.action
        return explanation

    return explain_kpi


def _make_get_kpi_snapshot(store):
    def get_kpi_snapshot() -> dict:
        return {"found": True, "snapshot": snapshot(store)}

    return get_kpi_snapshot


def _make_get_kpi_history(store):
    def get_kpi_history(days: int = 30) -> dict:
        rows = kpi_history(store, days=days)["days"]
        return {"found": True, "days": rows}

    return get_kpi_history


def _make_list_runs(store):
    def list_runs(limit: int = 20) -> dict:
        runs = store.list_runs(limit=limit)
        return {"found": True, "runs": [
            {
                "run_id": row["run_id"],
                "source": row["source_path"],
                "status": row["status"],
                "cost_usd": row["cost_usd"],
                "duration_ms": row["duration_ms"],
                "started_at": row["started_at"],
            }
            for row in runs
        ]}

    return list_runs


def _make_get_run_findings(store):
    def get_run_findings(run_id: str, verdict: str | None = None, quote_verified: bool | None = None) -> dict:
        rows = store.db.execute(
            "SELECT * FROM findings WHERE run_id = ? ORDER BY question_id, sub_requirement_id",
            (run_id,),
        ).fetchall()
        if not rows:
            return {"found": False, "run_id": run_id, "findings": [], "chunk_id": None}
        findings = []
        for row in rows:
            if verdict is not None and row["verdict"] != verdict:
                continue
            if quote_verified is not None and bool(row["quote_verified"]) != quote_verified:
                continue
            findings.append({
                "question_id": row["question_id"],
                "sub_requirement_id": row["sub_requirement_id"],
                "verdict": row["verdict"],
                "quote": row["quote"],
                "source_chunk_id": row["source_chunk_id"],
                "quote_verified": bool(row["quote_verified"]),
                "retry_count": row["retry_count"],
                "rationale": row["rationale"],
            })
        return {"found": True, "run_id": run_id, "findings": findings}

    return get_run_findings


def _make_get_compliance_result(store, *, jobs=None):
    def get_compliance_result(run_id: str, question_id: str | None = None) -> dict:
        if jobs is not None:
            job = jobs.get(run_id)
            if job is not None and job["status"] in {"running", "failed"}:
                return {"run_id": run_id, **job}
        result = store.get_run(run_id)
        if result is None:
            return _not_found(run_id=run_id, code="RUN_NOT_FOUND", error="No stored or active run has this ID.")
        questions = result.questions
        if question_id:
            questions = [q for q in questions if q.question_id == question_id]
            if not questions:
                return _not_found(run_id=run_id, question_id=question_id,
                                  code="QUESTION_NOT_FOUND", error="No question has this ID in the run.")
        return to_output_document(replace(result, questions=questions), None)

    return get_compliance_result


SEARCH_CONTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "k": {"type": "integer", "default": 6},
        "kinds": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["query"],
}

GET_SECTION_SCHEMA = {
    "type": "object",
    "properties": {"section_id": {"type": "string"}},
    "required": ["section_id"],
}

GET_REQUIREMENT_SCHEMA = {
    "type": "object",
    "properties": {"requirement_id": {"type": "string"}},
    "required": ["requirement_id"],
}

LIST_DEVIATIONS_SCHEMA = {"type": "object", "properties": {}}

GET_COMPLIANCE_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "run_id": {"type": "string"},
        "question_id": {"type": "string"},
    },
    "required": ["run_id"],
}

EXPLAIN_KPI_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
}

GET_KPI_SNAPSHOT_SCHEMA = {"type": "object", "properties": {}}

GET_KPI_HISTORY_SCHEMA = {
    "type": "object",
    "properties": {"days": {"type": "integer", "default": 30}},
}

LIST_RUNS_SCHEMA = {
    "type": "object",
    "properties": {"limit": {"type": "integer", "default": 20}},
}

GET_RUN_FINDINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "run_id": {"type": "string"},
        "verdict": {"type": "string"},
        "quote_verified": {"type": "boolean"},
    },
    "required": ["run_id"],
}


ANALYZE_CONTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "pdf_path": {"type": "string", "description": "Path readable by the server process."},
        "pdf_bytes": {"type": "string", "description": "Base64-encoded PDF bytes."},
        "allow_remote": {"type": "boolean", "default": False, "description": "Explicit authorization for non-confidential input only."},
        "assessment_date": {"type": "string", "description": "ISO date; defaults to the current UTC date."},
    },
    "oneOf": [{"required": ["pdf_path"]}, {"required": ["pdf_bytes"]}],
    "additionalProperties": False,
}


class _MCPJobs:
    """Owns asynchronous MCP runs and explicitly run-scoped in-memory indexes."""

    def __init__(self, store: Any, client: Any) -> None:
        self.store = store
        self.client = client
        self.jobs: dict[str, dict[str, Any]] = {}
        self.indexes: dict[str, ChunkIndex] = {}
        self._lock = threading.Lock()

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self.jobs.get(run_id)
            return dict(job) if job is not None else None

    def get_index(self, run_id: str) -> ChunkIndex | None:
        with self._lock:
            return self.indexes.get(run_id)

    def submit(self, pdf_path: str | None = None, pdf_bytes: str | None = None, *,
               allow_remote: bool = False, assessment_date: str | None = None) -> dict:
        if (pdf_path is None) == (pdf_bytes is None) or not (pdf_path or pdf_bytes):
            return _not_found(code="INVALID_ARGUMENTS", error="Provide exactly one of pdf_path or pdf_bytes.")
        run_id = str(uuid.uuid4())
        with self._lock:
            self.jobs[run_id] = {"status": "running"}
        thread = threading.Thread(
            target=self._run,
            args=(run_id, pdf_path, pdf_bytes, allow_remote, assessment_date),
            daemon=True,
            name=f"contract-analysis-{run_id[:8]}",
        )
        thread.start()
        return {"run_id": run_id, "status": "running"}

    def _run(self, requested_id: str, pdf_path: str | None, pdf_bytes: str | None,
             allow_remote: bool = False, assessment_date: str | None = None) -> None:
        temp_path = None
        try:
            if pdf_bytes:
                raw = base64.b64decode(pdf_bytes, validate=True)
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
                    handle.write(raw)
                    temp_path = Path(handle.name)
                path = temp_path
            else:
                path = Path(pdf_path)
                if not path.is_file():
                    raise FileNotFoundError(str(path))
            from core.ingest.parse import parse_pdf
            from core.providers.progress import backend_wait_updates

            def progress(event):
                with self._lock:
                    if event.phase == "resumed":
                        self.jobs[requested_id].pop("wait", None)
                    else:
                        self.jobs[requested_id]["wait"] = asdict(event)

            with span("mcp:analyze", run_id=requested_id), backend_wait_updates(progress):
                doc = parse_pdf(path)
                index = ChunkIndex(doc, allow_remote=allow_remote)
                result = analyze(path, self.client, index=index, allow_remote=allow_remote,
                                 assessment_date=assessment_date, run_id=requested_id)
                if result.run_id != requested_id:
                    raise ValueError("Analysis returned a different run_id")
            # RunStore owns a thread-bound sqlite connection; the worker opens
            # its own connection while readers continue using the MCP thread.
            from core.store import RunStore

            saved_store = RunStore(self.store.path)
            saved_store.save_run(result)
            with self._lock:
                self.indexes[requested_id] = index
                self.jobs[requested_id] = {"status": "incomplete" if result.incomplete else "completed"}
        except Exception as exc:
            from core.providers.errors import BackendError
            code = "BACKEND_UNAVAILABLE" if isinstance(exc, BackendError) else "ANALYSIS_FAILED"
            get_logger().error("mcp:analysis_failed", extra={"fields": {
                "run_id": requested_id, "code": code, "error_type": type(exc).__name__, "error": str(exc),
            }})
            with self._lock:
                self.jobs[requested_id] = {"status": "failed", "code": code, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


def _make_analyze_contract(jobs: _MCPJobs):
    def analyze_contract(pdf_path: str | None = None, pdf_bytes: str | None = None,
                         allow_remote: bool = False, assessment_date: str | None = None) -> dict:
        return jobs.submit(pdf_path=pdf_path, pdf_bytes=pdf_bytes,
                           allow_remote=allow_remote, assessment_date=assessment_date)

    return analyze_contract


def build_tool_registry(index, store, *, jobs=None, include_analyze=False, include_kpi=True) -> ToolRegistry:
    """Build the single registry the chat agent calls and task 08's MCP server wraps."""
    registry = ToolRegistry()
    evidence_registry = build_evidence_registry(index)
    for name in evidence_registry.names():
        registry.register(evidence_registry.get(name))
    registry.register(Tool(
        "get_audit_trail", "Read the local LangGraph execution audit for a stored run; no hidden model reasoning.",
        {"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"]},
        lambda run_id: {"found": store.get_run(run_id) is not None, "run_id": run_id,
                        "events": store.audit_trail(run_id)},
    ))
    registry.register(Tool(
        "search_contract",
        "Search the contract for chunks relevant to a query; returns ranked chunks with chunk_ids and citations.",
        SEARCH_CONTRACT_SCHEMA,
        _make_search_contract(index),
    ))
    registry.register(Tool(
        "get_section",
        "Fetch a specific contract clause chunk by its section id, e.g. '6.6'.",
        GET_SECTION_SCHEMA,
        _make_get_section(index),
    ))
    registry.register(Tool(
        "get_requirement",
        "Fetch a specific exhibit row by its requirement id, e.g. 'GOV-04'.",
        GET_REQUIREMENT_SCHEMA,
        _make_get_requirement(index),
    ))
    registry.register(Tool(
        "list_deviations",
        "List document-wide modifier candidates with source IDs and observed status; these are not automatic compliance overrides.",
        LIST_DEVIATIONS_SCHEMA,
        _make_list_deviations(index),
    ))
    registry.register(Tool(
        "get_compliance_result",
        "Poll running/waiting/failed analysis or read the shared public result with verified quotes, check_results and review_required.",
        GET_COMPLIANCE_RESULT_SCHEMA,
        _make_get_compliance_result(store, jobs=jobs),
    ))
    if include_kpi:
        registry.register(Tool(
            "explain_kpi",
            "Explain a KPI's definition, threshold, current value, and the action taken when it breaches the threshold.",
            EXPLAIN_KPI_SCHEMA,
            _make_explain_kpi(store),
        ))
        registry.register(Tool(
            "get_kpi_snapshot",
            "Return the current operational KPI snapshot derived from the run store.",
            GET_KPI_SNAPSHOT_SCHEMA,
            _make_get_kpi_snapshot(store),
        ))
        registry.register(Tool(
            "get_kpi_history",
            "Return daily KPI aggregates for historical trend questions.",
            GET_KPI_HISTORY_SCHEMA,
            _make_get_kpi_history(store),
        ))
        registry.register(Tool(
            "list_runs",
            "List the most recent runs along with source, status, cost, and duration metadata.",
            LIST_RUNS_SCHEMA,
            _make_list_runs(store),
        ))
        registry.register(Tool(
            "get_run_findings",
            "Fetch stored sub-requirement findings for a run, optionally filtered by verdict or quote verification.",
            GET_RUN_FINDINGS_SCHEMA,
            _make_get_run_findings(store),
        ))
    if include_analyze:
        if jobs is None:
            raise ValueError("jobs is required when include_analyze=True")
        registry.register(Tool(
            "analyze_contract",
            "Start asynchronous compliance analysis for a PDF; poll get_compliance_result with the returned run_id.",
            ANALYZE_CONTRACT_SCHEMA,
            _make_analyze_contract(jobs),
        ))
    return registry


def build_mcp_registry(store: Any | None = None, client: Any | None = None) -> ToolRegistry:
    """Construct the MCP-facing registry without duplicating tool definitions."""
    from core.store import RunStore

    store = store or RunStore(os.getenv("CA_RUN_DB", "data/runs.db"))
    client = client or LLMClient(ProviderConfig.from_env("llm"))
    jobs = _MCPJobs(store, client)
    registry = build_tool_registry(None, store, jobs=jobs, include_analyze=True, include_kpi=True)
    source_tools = [*build_evidence_registry(None).names(),
                    "search_contract", "get_section", "get_requirement", "list_deviations"]
    for name in source_tools:
        tool = registry.get(name)
        parameters = dict(tool.parameters,
                          properties={"run_id": {"type": "string", "description": "Exact run whose source to query."},
                                      **tool.parameters.get("properties", {})},
                          required=["run_id", *tool.parameters.get("required", [])],
                          additionalProperties=False)

        def scoped_handler(run_id: str, _name=name, **kwargs):
            current = jobs.get_index(run_id)
            if current is None:
                job = jobs.get(run_id)
                if job is not None and job["status"] in {"running", "failed"}:
                    return _not_found(run_id=run_id, **{
                        **job, "code": job.get("code", "RUN_NOT_READY"),
                        "error": job.get("error", "Analysis has not made its source available."),
                    })
                if store.get_run(run_id) is None:
                    return _not_found(run_id=run_id, code="RUN_NOT_FOUND", error="No run has this ID.")
                return _source_unavailable(run_id=run_id, code="SOURCE_UNAVAILABLE",
                                           error="This run has no live source index; stored results remain available.")
            result = build_tool_registry(current, store, include_kpi=False).call(_name, kwargs)
            return {**result, "run_id": run_id}

        registry.register(Tool(tool.name, tool.description + " MCP requires an explicit run_id.",
                               parameters, scoped_handler))
    return registry
