import os

import pytest

from core.providers.config import ProviderConfig
from core.providers.openai_compat import OpenAICompatClient


@pytest.mark.parametrize(
    ("backend", "base_url", "model"),
    [
        ("ollama", "http://localhost:11434/v1", "nomic-embed-text"),
        ("lmstudio", "http://localhost:1234/v1", "text-embedding-nomic-embed-text-v1.5"),
        ("openai", "https://api.openai.com/v1", "text-embedding-3-small"),
        ("groq", "https://api.groq.com/openai/v1", "nomic-embed-text"),
        ("siliconflow", "https://api.siliconflow.com/v1", "BAAI/bge-m3"),
    ],
)
def test_provider_config_defaults(backend, base_url, model):
    cfg = ProviderConfig(backend=backend, base_url=base_url, model=model)
    assert cfg.backend == backend
    assert cfg.base_url == base_url
    assert cfg.model == model
    assert cfg.api_key is None


@pytest.mark.parametrize(
    ("base_url", "provider"),
    [
        ("https://api.groq.com/openai/v1", "groq"),
        ("https://api.siliconflow.cn/v1", "siliconflow"),
        ("https://openrouter.ai/api/v1", "openrouter"),
        ("https://api.openai.com/v1", "openai"),
        ("http://localhost:11434/v1", "ollama"),
        ("http://localhost:1234/v1", "lmstudio"),
        ("https://example.com/v1", "unknown"),
    ],
)
def test_provider_config_provider_is_derived_from_base_url_not_backend(base_url, provider):
    cfg = ProviderConfig(backend="openai", base_url=base_url, model="model")
    assert cfg.backend == "openai"
    assert cfg.provider == provider


def test_provider_config_from_env_defaults(monkeypatch):
    monkeypatch.delenv("CA_EMBEDDING_BACKEND", raising=False)
    monkeypatch.delenv("CA_EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("CA_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("CA_EMBEDDING_API_KEY", raising=False)
    cfg = ProviderConfig.from_env("embedding")
    assert cfg.backend == "ollama"
    assert cfg.base_url == "http://localhost:11434/v1"
    assert cfg.model == "nomic-embed-text"
    assert cfg.api_key is None


def test_provider_config_from_env_overrides(monkeypatch):
    monkeypatch.setenv("CA_EMBEDDING_BACKEND", "siliconflow")
    monkeypatch.setenv("CA_EMBEDDING_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("CA_EMBEDDING_MODEL", "BAAI/bge-m3")
    monkeypatch.setenv("CA_EMBEDDING_API_KEY", "secret")
    cfg = ProviderConfig.from_env("embedding")
    assert cfg.backend == "siliconflow"
    assert cfg.base_url == "https://example.com/v1"
    assert cfg.model == "BAAI/bge-m3"
    assert cfg.api_key == "secret"


def test_fastembed_is_local_and_embedding_only(monkeypatch):
    for suffix in ("BASE_URL", "MODEL", "API_KEY"):
        monkeypatch.delenv(f"CA_EMBEDDING_{suffix}", raising=False)
    monkeypatch.setenv("CA_EMBEDDING_BACKEND", "fastembed")
    cfg = ProviderConfig.from_env("embedding")
    assert cfg.provider == "fastembed" and cfg.is_local
    assert cfg.model == "BAAI/bge-small-en-v1.5"
    monkeypatch.setenv("CA_LLM_BACKEND", "fastembed")
    with pytest.raises(ValueError, match="embedding-only"):
        ProviderConfig.from_env("llm")


def test_headers_include_bearer_authorization_and_user_agent():
    # Z1 (CRITICAL): the built Authorization header must carry the real key,
    # not a redacted placeholder -- a regression here silently disables
    # authentication against every hosted backend.
    # Z2: an explicit User-Agent is required because the default urllib UA is
    # rejected with 403 by Cloudflare, which fronts most hosted providers.
    cfg = ProviderConfig(backend="openai", base_url="https://example.com/v1", model="gpt-4o-mini", api_key="sk-test-synthetic-key")
    client = OpenAICompatClient(cfg)
    headers = client._headers()
    assert headers["Authorization"] == "Bearer sk-test-synthetic-key"
    assert headers["User-Agent"] == "contract-analyzer/0.1"


@pytest.mark.integration
def test_real_backend_roundtrip():
    from core.retrieval.embeddings import Embedder
    cfg = ProviderConfig.from_env("embedding")
    try:
        result = Embedder(cfg).encode(["password rotation"])
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"No embedding backend reachable: {exc}")
    assert len(result) == 1
    assert len(result[0]) > 0
