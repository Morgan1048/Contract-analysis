import math
from unittest.mock import patch

import pytest

from core.providers.config import ProviderConfig
from core.retrieval.embeddings import Embedder, FakeEmbedder, cosine_similarity
from core.retrieval.vectorstore import VectorStore


def test_embedder_identity_derives_dimension_from_real_vector():
    # Z3: a model that actually returns 1024-dim vectors (e.g. BAAI/bge-m3)
    # must report 1024 in its identity, never an assumed constant -- the
    # Chroma collection key is built from identity precisely so two different
    # embedding spaces cannot silently share a collection.
    cfg = ProviderConfig(backend="openai", base_url="https://example.com/v1", model="BAAI/bge-m3")
    embedder = Embedder(cfg)
    with patch.object(embedder._client, "embed", return_value=[[0.0] * 1024]) as mocked:
        identity = embedder.identity
        assert identity == "openai:BAAI/bge-m3:1024"
        mocked.assert_called_once()
        # Cached: accessing identity again must not re-encode.
        assert embedder.identity == identity
        mocked.assert_called_once()


def test_fake_embedder_dimensions_and_determinism():
    embedder = FakeEmbedder()
    first = embedder.encode(["password rotation", "credential rotation"])
    second = embedder.encode(["password rotation", "credential rotation"])
    assert len(first) == 2
    assert len(first[0]) == 64
    assert first == second


def test_embedder_identity_changes_with_backend_or_model():
    a = FakeEmbedder(model="model-a")
    b = FakeEmbedder(model="model-b")
    c = FakeEmbedder(model="model-a")
    assert a.identity != b.identity
    assert a.identity == c.identity


def test_vectorstores_do_not_share_collection_when_embedder_identity_differs():
    store_a = VectorStore(FakeEmbedder(model="embed-1"), persist_dir=".index/test-embeddings-a")
    store_b = VectorStore(FakeEmbedder(model="embed-2"), persist_dir=".index/test-embeddings-b")
    doc_hash = "abc123"
    assert store_a._collection_name(doc_hash) != store_b._collection_name(doc_hash)


@pytest.mark.integration
def test_real_backend_similarity_heuristic():
    try:
        cfg = ProviderConfig.from_env("embedding")
        vec_a, vec_b, vec_c = Embedder(cfg).encode(["password rotation", "credential rotation", "fire drill"])
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"No embedding backend reachable: {exc}")
    assert cosine_similarity(vec_a, vec_b) > cosine_similarity(vec_a, vec_c)


def test_fastembed_uses_real_local_adapter_without_http(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    calls = []

    class Vector:
        def tolist(self):
            return [1.0, 0.0, 0.5]

    class LocalModel:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def embed(self, texts):
            calls.append(texts)
            return [Vector() for _ in texts]

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=LocalModel))
    monkeypatch.chdir(tmp_path)
    cfg = ProviderConfig("fastembed", "local://fastembed", "BAAI/bge-small-en-v1.5")
    embedder = Embedder(cfg)
    assert embedder._client is None
    assert embedder.encode(["independent test"]) == [[1.0, 0.0, 0.5]]
    assert embedder.identity == "fastembed:BAAI/bge-small-en-v1.5:3"
    assert embedder.cost_usd == 0 and embedder.usage_estimated
    assert calls == [
        {"model_name": cfg.model, "cache_dir": ".index/models", "threads": 2},
        ["independent test"],
    ]
