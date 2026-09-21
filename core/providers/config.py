from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlparse

from dotenv import load_dotenv

# Z4: load .env once at import time, before any os.getenv() call below.
# override=False (the default) means real environment variables always win
# over whatever is in .env -- shell exports and deployment env stay
# authoritative. Values are never logged here or anywhere else.
load_dotenv()

BackendName = Literal["openai", "ollama", "lmstudio", "groq", "siliconflow", "fastembed"]
ProviderName = Literal["openai", "ollama", "lmstudio", "groq", "siliconflow", "openrouter", "fastembed", "unknown"]
RoleName = Literal["embedding", "llm"]

# Transport-layer defaults (X2): retry/backoff/timeout/pacing. Deliberately
# generous so a single rate-limited key degrades into visible retries and,
# eventually, a loud incomplete run rather than a fast, silent failure.
_TRANSPORT_DEFAULTS = {
    "timeout_s": 30.0,
    "max_retries": 12,
    "min_request_interval_s": 0.0,
    "backoff_base_s": 0.5,
    "backoff_cap_s": 20.0,
}

_DEFAULTS = {
    "fastembed": {
        "base_url": "local://fastembed",
        "embedding_model": "BAAI/bge-small-en-v1.5",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "embedding_model": "text-embedding-3-small",
        "llm_model": "gpt-4o-mini",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "embedding_model": "nomic-embed-text",
        "llm_model": "qwen2.5:14b-instruct",
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "embedding_model": "text-embedding-nomic-embed-text-v1.5",
        "llm_model": "local-model",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "embedding_model": "nomic-embed-text",
        "llm_model": "openai/gpt-oss-120b",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.com/v1",
        "embedding_model": "BAAI/bge-m3",
        "llm_model": "Qwen/Qwen3.5-9B",
    },
}


def _provider_from_base_url(base_url: str) -> ProviderName:
    parsed = urlparse(base_url)
    if parsed.hostname is None and "://" not in base_url:
        parsed = urlparse(f"//{base_url}")
    host = (parsed.hostname or "").lower()
    port = parsed.port

    if host == "api.groq.com":
        return "groq"
    if host in {"api.siliconflow.cn", "api.siliconflow.com"}:
        return "siliconflow"
    if host == "openrouter.ai":
        return "openrouter"
    if host == "api.openai.com":
        return "openai"
    if host in {"localhost", "127.0.0.1", "::1"} and port == 11434:
        return "ollama"
    if host in {"localhost", "127.0.0.1", "::1"} and port == 1234:
        return "lmstudio"
    return "unknown"


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class ProviderConfig:
    backend: BackendName
    base_url: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    # Transport-layer settings (X2): all configurable per role via env vars.
    timeout_s: float = _TRANSPORT_DEFAULTS["timeout_s"]
    max_retries: int = _TRANSPORT_DEFAULTS["max_retries"]
    min_request_interval_s: float = _TRANSPORT_DEFAULTS["min_request_interval_s"]
    backoff_base_s: float = _TRANSPORT_DEFAULTS["backoff_base_s"]
    backoff_cap_s: float = _TRANSPORT_DEFAULTS["backoff_cap_s"]
    max_input_tokens: int = 5500
    retry_budget_s: float = 1800.0
    rate_limit_backoff_s: float = 60.0

    @classmethod
    def from_env(cls, role: RoleName) -> "ProviderConfig":
        env_prefix = "CA_EMBEDDING" if role == "embedding" else "CA_LLM"
        backend_name = os.getenv(f"{env_prefix}_BACKEND", "ollama").strip().lower()
        if backend_name not in _DEFAULTS:
            raise ValueError(f"Unsupported {role} backend: {backend_name!r}")
        if backend_name == "fastembed" and role != "embedding":
            raise ValueError("fastembed is an embedding-only backend, not an LLM backend")

        defaults = _DEFAULTS[backend_name]
        base_url = os.getenv(f"{env_prefix}_BASE_URL", defaults["base_url"])
        model = os.getenv(f"{env_prefix}_MODEL", defaults["embedding_model" if role == "embedding" else "llm_model"])
        api_key = os.getenv(f"{env_prefix}_API_KEY")
        return cls(
            backend=backend_name,
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_s=_float_env(f"{env_prefix}_TIMEOUT_S", _TRANSPORT_DEFAULTS["timeout_s"]),
            max_retries=_int_env(f"{env_prefix}_MAX_RETRIES", _TRANSPORT_DEFAULTS["max_retries"]),
            min_request_interval_s=_float_env(
                f"{env_prefix}_MIN_INTERVAL_S", _TRANSPORT_DEFAULTS["min_request_interval_s"]
            ),
            backoff_base_s=_float_env(f"{env_prefix}_BACKOFF_BASE_S", _TRANSPORT_DEFAULTS["backoff_base_s"]),
            backoff_cap_s=_float_env(f"{env_prefix}_BACKOFF_CAP_S", _TRANSPORT_DEFAULTS["backoff_cap_s"]),
            max_input_tokens=_int_env(f"{env_prefix}_MAX_INPUT_TOKENS", 5500),
            retry_budget_s=_float_env(f"{env_prefix}_RETRY_BUDGET_S", 1800.0),
            rate_limit_backoff_s=_float_env(f"{env_prefix}_RATE_LIMIT_BACKOFF_S", 60.0),
        )

    @property
    def provider(self) -> ProviderName:
        if self.backend == "fastembed":
            return "fastembed"
        return _provider_from_base_url(self.base_url)

    @property
    def is_local(self) -> bool:
        return self.provider in {"ollama", "lmstudio", "fastembed"}
