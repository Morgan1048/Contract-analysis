from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from importlib.metadata import version

from core.agents.evaluator import Check
from core.agents.graph import Limits, analyze_question
from core.ingest.parse import parse_pdf
from core.obs.trace import current_trace_id, set_span_attributes, span
from core.retrieval.index import ChunkIndex
from core.rubric.aggregate import aggregate, load_rubric
from core.rubric.schema import QuestionResult


@dataclass(frozen=True)
class AnalysisResult:
    run_id: str
    source_path: str
    questions: list[QuestionResult]
    checks: dict[str, list[Check]]
    usage: dict
    started_at: str
    finished_at: str
    incomplete: bool = False
    audit_events: list[dict] = field(default_factory=list)
    assessment_date: str = ""
    document_hash: str = ""
    rubric_hash: str = ""
    evidence_snapshot: dict[str, dict] = field(default_factory=dict)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def analyze(pdf_path, client, rubric=None, index=None, *, assessment_date=None,
            limits: Limits | None = None, allow_remote: bool = False,
            run_id: str | None = None) -> AnalysisResult:
    """Shared entry point for the existing UI, chat tools, and MCP connector."""
    effective_date = date.fromisoformat(assessment_date) if assessment_date else datetime.now(timezone.utc).date()
    cfg = getattr(client, "cfg", None)
    if cfg is not None and not cfg.is_local and not allow_remote:
        raise ValueError("Remote inference requires explicit authorization for non-confidential input. Use a local model for confidential contracts.")
    limits = limits or Limits.from_env()
    if run_id is not None and not run_id.strip():
        raise ValueError("run_id must not be empty")
    started_at, run_id = _now(), run_id if run_id is not None else str(uuid.uuid4())
    timings, stage_usage, audit = {}, {}, []
    with span("analyze", run_id=run_id):
        run_trace_id = current_trace_id()
        t0 = time.perf_counter()
        with span("parse", run_id=run_id):
            doc = index.doc if index is not None else parse_pdf(pdf_path)
        timings["parse"] = max(0.001, (time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        with span("index", run_id=run_id):
            index = index or ChunkIndex(doc, allow_remote=allow_remote)
        timings["index"] = max(0.001, (time.perf_counter() - t0) * 1000)
        rubric = rubric or load_rubric()
        questions, checks, responses, evidence_snapshot, retrieved_sources = [], {}, [], {}, {}
        for question in rubric.values():
            state, calls = analyze_question(question, index, client, run_id=run_id,
                                            assessment_date=effective_date.isoformat(), limits=limits)
            audit.extend(state["events"])
            responses.extend(calls)
            checks.update(state["checks"])
            evidence_snapshot.update((c.chunk_id, json.loads(json.dumps(asdict(c))))
                                     for p in state["packs"].values() for c in p.chunks)
            retrieved_sources.update((sid, [c.chunk_id for c in pack.chunks]) for sid, pack in state["packs"].items())
            with span(f"aggregate:{question.id}", run_id=run_id, question_id=question.id):
                questions.append(aggregate(question, list(state["findings"].values())))
        incomplete = any(q.state == "Indeterminate" or any(not f.search_complete for f in q.findings)
                         for q in questions)
        set_span_attributes(output={q.question_id: q.state for q in questions}, incomplete=incomplete)
    for event in audit:
        stage = event["node"]
        timings[stage] = timings.get(stage, 0) + event["duration_ms"]
        usage = stage_usage.setdefault(stage, {"prompt_tokens": 0, "completion_tokens": 0, "tokens": 0, "cost_usd": 0.0})
        usage["prompt_tokens"] += event["prompt_tokens"]
        usage["completion_tokens"] += event["completion_tokens"]
        usage["tokens"] += event["prompt_tokens"] + event["completion_tokens"]
        usage["cost_usd"] = (None if usage["cost_usd"] is None or event["cost_usd"] is None
                             else usage["cost_usd"] + event["cost_usd"])
    cfg = getattr(client, "cfg", None)
    prompt_tokens = sum(r.prompt_tokens for r in responses)
    completion_tokens = sum(r.completion_tokens for r in responses)
    quotes = [a for e in audit for a in e["quote_attempts"]]
    retrieved_again = {(e["question_id"], s) for e in audit if e["node"] == "router" and e["round"] > 0 for s in e["pending"]}
    resolved_again = {(q.question_id, f.sub_requirement_id) for q in questions for f in q.findings
                      if f.search_complete and (q.question_id, f.sub_requirement_id) in retrieved_again}
    embedding_usage = index.consume_embedding_usage()
    unaccounted = sum(e["unaccounted_llm_calls"] for e in audit)
    unpriced = any(r.cost_usd is None for r in responses) or embedding_usage["cost_usd"] is None or unaccounted > 0
    llm_cost = None if any(r.cost_usd is None for r in responses) else sum(r.cost_usd for r in responses)
    stage_usage["embedding"] = {"tokens": embedding_usage["prompt_tokens"], "cost_usd": embedding_usage["cost_usd"]}
    return AnalysisResult(
        run_id, str(pdf_path), questions, checks,
        {
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens + embedding_usage["prompt_tokens"],
            "retries": sum(f.retry_count for q in questions for f in q.findings),
            "latency_ms": timings, "stages": timings, "stage_usage": stage_usage,
            "cost_usd": None if unpriced else llm_cost + embedding_usage["cost_usd"],
            "llm_cost_usd": llm_cost, "embedding_usage": embedding_usage,
            "cost_basis": "Configured provider price table, including embeddings; not an invoice. Groq entries assume free-tier API use.",
            "unaccounted_llm_calls": unaccounted,
            "unpriced_model": unpriced, "usage_estimated": any(r.usage_estimated for r in responses),
            "model": getattr(client, "model", getattr(cfg, "model", "unknown")),
            "backend": getattr(client, "backend", getattr(cfg, "backend", "local")),
            "provider": getattr(cfg, "provider", "local"),
            "langgraph_version": version("langgraph"),
            "rubric_version": 2,
            "embedded_images": doc.embedded_image_count,
            "chunk_count": len(doc.chunks), "table_row_chunks": sum(c.kind == "table_row" for c in doc.chunks),
            "llm_calls": sum(e["llm_calls"] for e in audit),
            "tool_calls": sum(len(e["tool_calls"]) for e in audit),
            "quote_attempts": len(quotes), "quote_failures": sum(not a["passed"] for a in quotes),
            "followup_checks": len(retrieved_again), "resolved_followup_checks": len(resolved_again),
            "budget_exhausted": any("budget" in e["stop_reason"] for e in audit),
            "parse_warnings": doc.warnings, "retrieval_warning": index.degraded_reason,
            "confidence_method": "heuristic: minimum of verified judge and critic confidence; not statistically calibrated",
            "pipeline_version": "langgraph-evidence-v2",
            "question_timeout_s": limits.seconds,
            "trace_id": run_trace_id,
            "allow_remote": allow_remote,
            "retrieved_sources": retrieved_sources,
        },
        started_at, _now(), incomplete,
        audit, effective_date.isoformat(),
        hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest() if Path(pdf_path).is_file()
        else hashlib.sha256(doc.text.encode()).hexdigest(),
        hashlib.sha256(json.dumps([asdict(q) for q in rubric.values()], sort_keys=True).encode()).hexdigest(),
        evidence_snapshot,
    )
