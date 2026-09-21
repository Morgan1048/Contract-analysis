"""Local-only Langfuse configuration; no network activity during UI rendering."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from urllib.parse import quote, urlsplit

from dotenv import dotenv_values


@dataclass(frozen=True)
class LangfuseConfig:
    base_url: str
    project_id: str
    public_key: str = field(repr=False)
    secret_key: str = field(repr=False)

    @property
    def dashboard_url(self) -> str:
        return f"{self.base_url}/project/{quote(self.project_id, safe='')}"

    def trace_url(self, trace_id: str) -> str:
        return f"{self.dashboard_url}/traces/{quote(trace_id, safe='')}"


def config() -> LangfuseConfig | None:
    local = dotenv_values(Path(__file__).resolve().parents[2] / ".env.langfuse")

    def value(name, default=""):
        return os.environ.get(name, local.get(name) or default)

    if value("CA_LANGFUSE_ENABLED", "0").lower() not in {"1", "true", "yes", "on"}:
        return None
    base = value("LANGFUSE_BASE_URL", "http://127.0.0.1:3100").rstrip("/")
    parsed = urlsplit(base)
    if (parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path):
        raise ValueError("Langfuse must use a loopback HTTP(S) URL; hosted export is disabled for confidential evidence.")
    public, secret = value("LANGFUSE_PUBLIC_KEY"), value("LANGFUSE_SECRET_KEY")
    project = value("LANGFUSE_PROJECT_ID")
    if not public or not secret or not project:
        raise ValueError("Langfuse requires project ID, public key and secret key. Run ./setup.sh --langfuse.")
    return LangfuseConfig(base, project, public, secret)
