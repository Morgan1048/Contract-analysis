from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable

from core.providers.config import ProviderConfig
from core.providers.openai_compat import OpenAICompatClient
from core.providers.errors import BackendError
from core.pricing import cost_usd
from core.obs.trace import span, set_span_attributes


@dataclass(frozen=True)
class EmbeddingVector:
    values: list[float]


class Embedder:
    def __init__(self, cfg: ProviderConfig) -> None:
        self.cfg = cfg
        self._local_model = None
        self._client = None
        if cfg.backend == "fastembed":
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise BackendError("Local embeddings require fastembed; run setup.sh --install-only.") from exc
            self._local_model = TextEmbedding(model_name=cfg.model, cache_dir=".index/models", threads=2)
        else:
            self._client = OpenAICompatClient(cfg)
        self._dim: int | None = None
        self.calls = 0
        self.prompt_tokens = 0
        self.cost_usd: float | None = 0.0
        self.usage_estimated = False

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with span("llm:embed", observation_type="embedding", model=self.cfg.model,
                  system=self.cfg.provider, input={"texts": texts}):
            return self._encode(texts)

    def _encode(self, texts):
        self.calls += 1
        if self._local_model is not None:
            vectors = [vector.tolist() for vector in self._local_model.embed(texts)]
        elif self._client is not None:
            vectors = self._client.embed(texts)
        else:
            raise BackendError("No embedding backend initialized")
        if not vectors:
            raise RuntimeError("Embedding backend returned no vectors")
        self._dim = len(vectors[0])
        usage = getattr(self._client, "last_embedding_usage", {})
        tokens = usage.get("prompt_tokens", usage.get("total_tokens"))
        if tokens is None:
            self.usage_estimated = True
            tokens = max(1, sum(len(text) for text in texts) // 4)
        self.prompt_tokens += tokens
        price = cost_usd(self.cfg.model, tokens, 0, is_local=self.cfg.is_local, provider=self.cfg.provider)
        self.cost_usd = None if price is None or self.cost_usd is None else self.cost_usd + price
        set_span_attributes(prompt_tokens=tokens, completion_tokens=0, cost_usd=price,
                            usage_estimated=usage.get("prompt_tokens", usage.get("total_tokens")) is None,
                            output={"vectors": len(vectors), "dimensions": self._dim})
        return vectors

    @property
    def identity(self) -> str:
        # Z3: the dimension must come from a real encoded vector, never an
        # assumed constant -- the Chroma collection key is built from this
        # identity precisely so two different embedding spaces cannot share a
        # collection, and a guessed dimension defeats that silently.
        if self._dim is None:
            self.encode(["dimension probe"])
        return f"{self.cfg.backend}:{self.cfg.model}:{self._dim}"


class FakeEmbedder:
    """Deterministic hash-based 64-dim vectors. Unit tests use this so the suite runs with no
    server and no key; it is never used outside tests."""

    def __init__(self, cfg: ProviderConfig | None = None, model: str = "fake-embedder") -> None:
        self.cfg = cfg or ProviderConfig(backend="ollama", base_url="http://localhost:11434/v1", model=model)
        self.model = model
        self.dim = 64

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values: list[float] = []
            for idx in range(self.dim):
                byte = digest[idx % len(digest)]
                slope = (byte / 255.0) * 2.0 - 1.0
                values.append(round(slope + (idx / self.dim), 6))
            vectors.append(values)
        return vectors

    @property
    def identity(self) -> str:
        return f"{self.cfg.backend}:{self.model}:{self.dim}"


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    left = list(a)
    right = list(b)
    if not left or not right:
        return 0.0
    dot = sum(x * y for x, y in zip(left, right))
    norm_left = math.sqrt(sum(x * x for x in left))
    norm_right = math.sqrt(sum(x * x for x in right))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    return dot / (norm_left * norm_right)
