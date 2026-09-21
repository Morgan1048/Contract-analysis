from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from core.kpi import Alert, evaluate_thresholds
from core.ingest.parse import parse_pdf
from core.llm.client import LLMClient
from core.llm.client import LLMResponse
from core.llm.stub import StubLLM
from core.pipeline import analyze
from core.providers.config import ProviderConfig
from core.retrieval.embeddings import FakeEmbedder
from core.retrieval.index import ChunkIndex
from core.retrieval.vectorstore import VectorStore
from core.store import RunStore
from eval.variants import generate_variants

GOLD_PATH = Path(__file__).with_name("gold.yaml")


@dataclass(frozen=True)
class EvalReport:
    client_identity: str = ""
    backend: str = ""
    model: str = ""
    measurement_scope: str = ""
    threshold_policy: str = ""
    threshold_alerts: list[dict[str, str | float]] = field(default_factory=list)
    documents: int = 0
    question_accuracy: dict[str, float] = field(default_factory=dict)
    sub_requirement_accuracy: float | None = None
    quote_verification_failure_rate: float = 0.0
    mean_confidence_by_correctness: dict[str, float | None] = field(default_factory=dict)
    calibration: list[dict[str, float | int]] = field(default_factory=list)
    per_document_cost: dict[str, float | None] = field(default_factory=dict)
    per_document_latency_ms: dict[str, float] = field(default_factory=dict)


class DeterministicEvalClient(StubLLM):
    """Offline orchestration fixture. Its outputs must never be scored as accuracy."""
    backend = "deterministic"

    def __init__(self):
        super().__init__(model="deterministic-eval-stub")


def load_gold(path: str | Path = GOLD_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _document_key(path: Path) -> str:
    if path.name == "Sample Contract.pdf":
        return "sample"
    return path.stem


def _empty_report() -> EvalReport:
    return EvalReport(
        threshold_policy="No thresholds evaluated because no matching fixtures were supplied.",
        mean_confidence_by_correctness={"correct": None, "incorrect": None},
        calibration=[],
    )


def _mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 3) if values else None


def _calibration(rows: list[tuple[float, bool]]) -> list[dict[str, float | int]]:
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    output: list[dict[str, float | int]] = []
    for lower, upper in bins:
        if upper == 1.0:
            in_bin = [(confidence, correct) for confidence, correct in rows if lower <= confidence <= upper]
        else:
            in_bin = [(confidence, correct) for confidence, correct in rows if lower <= confidence < upper]
        if not in_bin:
            output.append({"lower": lower, "upper": upper, "count": 0, "accuracy": 0.0, "mean_confidence": 0.0})
            continue
        output.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(in_bin),
                "accuracy": round(sum(1 for _, correct in in_bin if correct) / len(in_bin), 3),
                "mean_confidence": round(mean(confidence for confidence, _ in in_bin), 3),
            }
        )
    return output


def _insert_eval_row(store: RunStore, run_id: str, question_id: str, expected: str, actual: str, correct: bool) -> None:
    with store.db:
        store.db.execute(
            "INSERT INTO eval_results VALUES (?,?,?,?,?,?,?,?)",
            (
                str(uuid.uuid4()),
                run_id,
                question_id,
                expected,
                actual,
                int(correct),
                datetime.now(timezone.utc).isoformat(),
                0,
            ),
        )


def _client_metadata(client) -> tuple[str, str, str, str, str]:
    cfg = getattr(client, "cfg", None)
    identity = client.__class__.__name__
    backend = str(getattr(client, "backend", getattr(cfg, "backend", "unknown")))
    model = str(getattr(client, "model", getattr(cfg, "model", "unknown")))
    if isinstance(client, DeterministicEvalClient):
        scope = (
            "Deterministic fixture run: measures pipeline behaviour under changed evidence, "
            "not model judgment quality."
        )
        policy = (
            "Operational metrics only. Synthetic stub outputs are excluded from model accuracy and calibration."
        )
    else:
        scope = "Live backend run: measures the configured model plus the parser/router/evaluator pipeline."
        policy = "Thresholds are evaluated and any breach is fatal for the CLI harness."
    return identity, backend, model, scope, policy


def _alerts_to_dicts(alerts: list[Alert]) -> list[dict[str, str | float]]:
    return [
        {
            "name": alert.name,
            "value": round(alert.value, 6),
            "threshold": alert.threshold,
            "severity": alert.severity,
            "action": alert.action,
        }
        for alert in alerts
    ]


def _threshold_alerts(report: EvalReport) -> list[Alert]:
    latencies = list(report.per_document_latency_ms.values())
    priced_costs = [cost for cost in report.per_document_cost.values() if cost is not None]
    snapshot = {
        "runs": report.documents,
        "quote_verification_failure_rate": report.quote_verification_failure_rate,
        "not_found_rate": 0.0,
        "mean_retry_count": 0.0,
        "p95_latency_ms": max(latencies) if latencies else 0.0,
        "cost_per_document": (sum(priced_costs) / len(priced_costs)) if priced_costs else 0.0,
        "cost_median_7d": (sum(priced_costs) / len(priced_costs)) if priced_costs else 0.0,
        # These KPIs are not derived by the eval report; set neutral values so
        # applying thresholds here targets the metrics the report actually owns.
        "latest_chunk_count": 40,
        "latest_table_row_chunks": 1,
        "deviation_applied_rate": 1.0,
    }
    return evaluate_thresholds(
        snapshot,
        calibration_bins=None,
        accuracy=report.sub_requirement_accuracy if report.documents else None,
    )


def _with_thresholds(report: EvalReport) -> EvalReport:
    alerts = _threshold_alerts(report) if report.documents else []
    return EvalReport(
        client_identity=report.client_identity,
        backend=report.backend,
        model=report.model,
        measurement_scope=report.measurement_scope,
        threshold_policy=report.threshold_policy,
        threshold_alerts=_alerts_to_dicts(alerts),
        documents=report.documents,
        question_accuracy=report.question_accuracy,
        sub_requirement_accuracy=report.sub_requirement_accuracy,
        quote_verification_failure_rate=report.quote_verification_failure_rate,
        mean_confidence_by_correctness=report.mean_confidence_by_correctness,
        calibration=report.calibration,
        per_document_cost=report.per_document_cost,
        per_document_latency_ms=report.per_document_latency_ms,
    )


def _deterministic_index(fixture: Path) -> ChunkIndex:
    doc = parse_pdf(fixture)
    store = VectorStore(FakeEmbedder(), persist_dir=".index/eval-fake")
    return ChunkIndex(doc, store=store)


def run_eval(store: RunStore, client, fixtures: list[Path]) -> EvalReport:
    client = client or DeterministicEvalClient()
    identity, backend, model, scope, policy = _client_metadata(client)
    if not fixtures:
        empty = _empty_report()
        return EvalReport(
            client_identity=identity,
            backend=backend,
            model=model,
            measurement_scope=scope,
            threshold_policy=empty.threshold_policy,
            threshold_alerts=[],
            documents=empty.documents,
            question_accuracy=empty.question_accuracy,
            sub_requirement_accuracy=empty.sub_requirement_accuracy,
            quote_verification_failure_rate=empty.quote_verification_failure_rate,
            mean_confidence_by_correctness=empty.mean_confidence_by_correctness,
            calibration=empty.calibration,
            per_document_cost=empty.per_document_cost,
            per_document_latency_ms=empty.per_document_latency_ms,
        )

    gold_payload = load_gold()
    if not isinstance(client, DeterministicEvalClient) and gold_payload.get("rubric_version") != 2:
        raise ValueError("Historical gold labels need independent review for rubric version 2 before live accuracy evaluation.")
    gold = gold_payload["documents"]
    by_question: dict[str, list[bool]] = {}
    sub_total = 0
    sub_correct = 0
    quote_total = 0
    quote_failures = 0
    confidence_rows: list[tuple[float, bool]] = []
    correct_confidences: list[float] = []
    incorrect_confidences: list[float] = []
    per_document_cost: dict[str, float | None] = {}
    per_document_latency_ms: dict[str, float] = {}

    for fixture in fixtures:
        fixture = Path(fixture)
        key = _document_key(fixture)
        expected_document = gold.get(key)
        if expected_document is None:
            continue
        index = _deterministic_index(fixture) if isinstance(client, DeterministicEvalClient) else None
        result = analyze(fixture, client, index=index)
        store.save_run(result)

        per_document_cost[key] = result.usage.get("cost_usd")
        per_document_latency_ms[key] = round(sum(result.usage.get("latency_ms", {}).values()), 3)
        if isinstance(client, DeterministicEvalClient):
            continue
        expected_questions = expected_document["questions"]
        for question in result.questions:
            expected = expected_questions.get(question.question_id)
            if expected is None:
                continue
            correct = question.state == expected["expected_state"]
            by_question.setdefault(question.question_id, []).append(correct)
            confidence_rows.append((question.confidence, correct))
            (correct_confidences if correct else incorrect_confidences).append(question.confidence)
            _insert_eval_row(store, result.run_id, question.question_id, expected["expected_state"], question.state, correct)

            expected_subs = expected.get("sub_requirements", {})
            findings = {finding.sub_requirement_id: finding for finding in question.findings}
            for sub_id, expected_verdict in expected_subs.items():
                sub_total += 1
                actual = findings.get(sub_id)
                if actual is not None and actual.verdict == expected_verdict:
                    sub_correct += 1
            for finding in question.findings:
                quote_total += 1
                if not finding.quote_verified:
                    quote_failures += 1

    if not by_question:
        return _with_thresholds(EvalReport(
            client_identity=identity,
            backend=backend,
            model=model,
            measurement_scope=scope,
            threshold_policy=policy,
            mean_confidence_by_correctness={"correct": None, "incorrect": None},
            calibration=[],
            documents=len(per_document_cost),
            per_document_cost=per_document_cost,
            per_document_latency_ms=per_document_latency_ms,
        ))
    return _with_thresholds(EvalReport(
        client_identity=identity,
        backend=backend,
        model=model,
        measurement_scope=scope,
        threshold_policy=policy,
        documents=len(fixtures),
        question_accuracy={
            question_id: round(sum(results) / len(results), 3)
            for question_id, results in sorted(by_question.items())
        },
        sub_requirement_accuracy=round(sub_correct / sub_total, 3) if sub_total else 0.0,
        quote_verification_failure_rate=round(quote_failures / quote_total, 3) if quote_total else 0.0,
        mean_confidence_by_correctness={
            "correct": _mean_or_none(correct_confidences),
            "incorrect": _mean_or_none(incorrect_confidences),
        },
        calibration=_calibration(confidence_rows),
        per_document_cost=per_document_cost,
        per_document_latency_ms=per_document_latency_ms,
    ))


def _client_from_args(client_name: str):
    if client_name == "deterministic":
        return DeterministicEvalClient()
    return LLMClient(ProviderConfig.from_env("llm"))


def _fatal_alerts(report: EvalReport) -> list[dict[str, str | float]]:
    if report.backend == "deterministic":
        return [
            alert for alert in report.threshold_alerts
            if alert.get("name") != "quote_verification_failure_rate"
        ]
    return report.threshold_alerts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the contract analyzer evaluation harness.")
    parser.add_argument(
        "--client",
        choices=("deterministic", "env"),
        default="deterministic",
        help="Use the offline deterministic fixture client, or the real LLM configured from .env/ENV.",
    )
    args = parser.parse_args(argv)
    variants = generate_variants()
    fixtures = [Path("Sample Contract.pdf"), *variants.values()]
    report = run_eval(RunStore(), _client_from_args(args.client), fixtures)
    print(json.dumps(asdict(report), indent=2))
    fatal_alerts = _fatal_alerts(report)
    if fatal_alerts:
        names = ", ".join(str(alert["name"]) for alert in fatal_alerts)
        raise SystemExit(f"Evaluation KPI thresholds breached: {names}")


if __name__ == "__main__":
    main(sys.argv[1:])
