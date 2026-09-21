import json
import sys

import pytest

from core.ingest.parse import parse_pdf
from core.llm.stub import StubLLM
from core.pipeline import analyze
from core.providers.config import ProviderConfig
from core.retrieval.embeddings import FakeEmbedder
from core.retrieval.index import ChunkIndex
from core.retrieval.vectorstore import VectorStore
from eval import live_e2e


@pytest.mark.parametrize("degraded", [False, True])
def test_alternate_provider_preserves_transport_settings_and_public_result(tmp_path, monkeypatch, degraded):
    config = ProviderConfig("openai", "https://api.siliconflow.cn/v1", "embedding-model",
                            max_retries=9, retry_budget_s=900, timeout_s=75)
    selected_roles, clients = [], []

    def configured(role):
        selected_roles.append(role)
        return config

    def client(cfg):
        clients.append(cfg)
        return StubLLM()

    def offline_analyze(path, llm, **kwargs):
        index = ChunkIndex(parse_pdf(path), VectorStore(FakeEmbedder(), tmp_path / "index"))
        if degraded:
            index.degraded_reason = "Embedding service unavailable; lexical-only retrieval."
        return analyze(path, llm, index=index, **kwargs)

    monkeypatch.setattr(live_e2e.ProviderConfig, "from_env", configured)
    monkeypatch.setattr(live_e2e, "LLMClient", client)
    monkeypatch.setattr(live_e2e, "analyze", offline_analyze)
    output = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", [
        "live_e2e", "--use-embedding-provider", "--model", "text-model",
        "--smoke", "--output-dir", str(output),
    ])
    if degraded:
        with pytest.raises(SystemExit, match="E2E assertions failed"):
            live_e2e.main()
    else:
        live_e2e.main()
    assert selected_roles == ["embedding"]
    assert clients[0].model == "text-model"
    assert clients[0].base_url == config.base_url
    assert clients[0].max_retries == 9
    assert clients[0].retry_budget_s == 900
    assert clients[0].timeout_s == 75
    report = json.loads((output / "report.json").read_text())
    assert report["cases"][0]["passed"] is not degraded
    assert report["provider"] == "siliconflow"
    assert report["model"] == "text-model"


def test_alternate_endpoint_requires_explicit_text_model(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["live_e2e", "--use-embedding-provider", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        live_e2e.main()
    assert error.value.code == 2


def test_native_embedding_provider_cannot_be_used_as_a_language_model(tmp_path, monkeypatch):
    cfg = ProviderConfig("fastembed", "local://fastembed", "BAAI/bge-small-en-v1.5")
    monkeypatch.setattr(live_e2e.ProviderConfig, "from_env", lambda role: cfg)
    monkeypatch.setattr(sys, "argv", [
        "live_e2e", "--use-embedding-provider", "--model", "not-a-text-endpoint",
        "--output-dir", str(tmp_path),
    ])
    with pytest.raises(SystemExit) as error:
        live_e2e.main()
    assert error.value.code == 2
