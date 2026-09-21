"""Explicit opt-in live-provider E2E over non-confidential synthetic material."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from jsonschema import validate

from core.kpi import snapshot
from core.llm.client import LLMClient
from core.openapi_schema import analysis_result_schema
from core.output import to_output_document
from core.pipeline import analyze
from core.providers.config import ProviderConfig
from core.store import RunStore
from core.tools.contract_tools import build_tool_registry
from eval.synthetic import rubric, write_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Run only one group before the full 25-check test.")
    parser.add_argument("--model", help="Optional model override on the same configured provider/key; never changes .env.")
    parser.add_argument("--use-embedding-provider", action="store_true",
                        help="Use the existing embedding endpoint/key for LLM calls; requires --model for a text model.")
    parser.add_argument("--case", choices=("complete", "missing_obligation"),
                        help="Run one fixture when refining a specific live failure.")
    args = parser.parse_args()
    if args.smoke and args.case == "missing_obligation":
        parser.error("--smoke cannot be combined with --case missing_obligation")
    if args.use_embedding_provider and not args.model:
        parser.error("--use-embedding-provider requires an explicit text --model")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = ProviderConfig.from_env("embedding" if args.use_embedding_provider else "llm")
    if config.backend == "fastembed":
        parser.error("fastembed serves local embeddings only; use the configured LLM endpoint for this test")
    if args.model:
        config = replace(config, model=args.model)
    client = LLMClient(config)
    store = RunStore(args.output_dir / "runs.db")
    requirements = rubric()
    if args.smoke:
        requirements = {"Q1": requirements["Q1"]}
    cases = [("complete", None, False, requirements)]
    if not args.smoke:
        cases.append(("missing_obligation", "moisture_barrier", True, {"Q5": requirements["Q5"]}))
    if args.case:
        cases = [case for case in cases if case[0] == args.case]
    report = {"provider": config.provider, "model": config.model, "scope": "independent non-confidential synthetic fixtures", "cases": []}
    failures = []
    for name, omit, renumber, checks in cases:
        pdf = write_contract(args.output_dir / f"{name}.pdf", omit=omit, renumber=renumber)
        result = analyze(pdf, client, rubric=checks, assessment_date="2026-01-15", allow_remote=True)
        store.save_run(result)
        restored = store.get_run(result.run_id)
        output = to_output_document(restored, None)
        validate(output, analysis_result_schema())
        registry = build_tool_registry(None, store)
        assert registry.call("get_audit_trail", {"run_id": result.run_id})["events"]
        assert registry.call("get_compliance_result", {"run_id": result.run_id}) == output
        expected = "Partially Compliant" if omit else "Fully Compliant"
        findings = [f for q in result.questions for f in q.findings]
        expected_ids = {sub.id for question in checks.values() for sub in question.sub_requirements}
        passed = (
            not result.incomplete and not result.usage.get("retrieval_warning")
            and {q.question_id for q in result.questions} == set(checks)
            and len(findings) == len(expected_ids)
            and {f.sub_requirement_id for f in findings} == expected_ids
            and all(q.state == expected for q in result.questions)
            and all(f.search_complete for f in findings)
        )
        if omit:
            passed = passed and next(f for f in findings if f.sub_requirement_id == omit).verdict == "NOT_FOUND"
        passed = passed and all(f.quote_verified for f in findings)
        record = {"case": name, "run_id": result.run_id, "passed": passed,
                  "states": {q.question_id: q.state for q in result.questions},
                  "findings": {f.sub_requirement_id: f.verdict for f in findings},
                  "usage": result.usage}
        report["cases"].append(record)
        (args.output_dir / f"{name}.json").write_text(json.dumps(output, indent=2))
        if not passed:
            failures.append(name)
    report["kpi"] = snapshot(store)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit("E2E assertions failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
