from __future__ import annotations

from pathlib import Path

import yaml

from core.ingest.parse import parse_pdf
from core.pipeline import analyze
from core.retrieval.embeddings import FakeEmbedder
from core.retrieval.index import ChunkIndex
from core.retrieval.vectorstore import VectorStore
from core.store import RunStore
from eval.run_eval import DeterministicEvalClient, _client_from_args, run_eval
from eval.variants import generate_variants


def test_gold_covers_all_questions_and_sub_requirements():
    gold = yaml.safe_load(Path("eval/gold.yaml").read_text(encoding="utf-8"))
    sample = gold["documents"]["sample"]["questions"]
    assert set(sample) == {"Q1", "Q2", "Q3", "Q4", "Q5"}
    assert sum(len(question["sub_requirements"]) for question in sample.values()) == 25


def test_variant_generator_produces_parseable_pdfs(tmp_path):
    paths = generate_variants("Sample Contract.pdf", tmp_path)
    assert set(paths) == {"no_tls_version", "no_training_row", "deviation_closed"}
    for path in paths.values():
        doc = parse_pdf(path)
        assert doc.page_count >= 1
        assert doc.chunks


def _fake_index(path: Path, tmp_path: Path) -> ChunkIndex:
    return ChunkIndex(parse_pdf(path), VectorStore(FakeEmbedder(), persist_dir=tmp_path / "fake-index"))


def test_no_tls_variant_removes_protocol_baseline_from_fixture(tmp_path):
    path = generate_variants("Sample Contract.pdf", tmp_path)["no_tls_version"]
    doc = parse_pdf(path)
    assert "TLS 1.2 or higher" not in doc.text


def test_training_variant_removes_target_rows_without_inventing_model_accuracy(tmp_path):
    path = generate_variants("Sample Contract.pdf", tmp_path)["no_training_row"]
    doc = parse_pdf(path)
    assert "GOV-04" not in doc.text
    assert "GOV-05" not in doc.text


def test_stub_exercises_graph_without_automatic_id_based_downgrades(tmp_path):
    path = generate_variants("Sample Contract.pdf", tmp_path)["deviation_closed"]
    result = analyze(path, DeterministicEvalClient(), index=_fake_index(path, tmp_path))
    by_question = {question.question_id: question for question in result.questions}
    assert by_question["Q1"].state == "Fully Compliant"
    assert by_question["Q5"].state == "Fully Compliant"
    assert by_question["Q1"].applied_deviations == []
    assert by_question["Q5"].applied_deviations == []


def test_run_eval_writes_results_and_computes_metrics(tmp_path):
    paths = generate_variants("Sample Contract.pdf", tmp_path / "fixtures")
    fixtures = [Path("Sample Contract.pdf"), *paths.values()]
    store = RunStore(tmp_path / "runs.db")
    report = run_eval(store, DeterministicEvalClient(), fixtures)
    assert report.client_identity == "DeterministicEvalClient"
    assert report.backend == "deterministic"
    assert report.model == "deterministic-eval-stub"
    assert "pipeline behaviour" in report.measurement_scope
    assert report.documents == 4
    assert report.question_accuracy == {}
    assert report.sub_requirement_accuracy is None
    assert report.calibration == []
    assert set(report.per_document_cost) == {"sample", "no_tls_version", "no_training_row", "deviation_closed"}
    row_count = store.db.execute("SELECT COUNT(*) FROM eval_results").fetchone()[0]
    assert row_count == 0


def test_run_eval_empty_fixture_list_returns_empty_report(tmp_path):
    report = run_eval(RunStore(tmp_path / "runs.db"), DeterministicEvalClient(), [])
    assert report.documents == 0
    assert report.client_identity == "DeterministicEvalClient"
    assert report.question_accuracy == {}
    assert report.calibration == []


def test_eval_env_client_uses_dotenv_provider_config(monkeypatch):
    monkeypatch.setenv("CA_LLM_BACKEND", "groq")
    monkeypatch.setenv("CA_LLM_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setenv("CA_LLM_API_KEY", "synthetic")
    client = _client_from_args("env")
    assert client.cfg.backend == "groq"
    assert client.cfg.model == "openai/gpt-oss-120b"
