"""Provision private local credentials once, then run the isolated Compose stack."""
from __future__ import annotations

import os
from pathlib import Path
import secrets

ROOT = Path(__file__).resolve().parents[1]


def initialize(path: Path) -> None:
    if path.exists():
        return
    values = {
        "CA_LANGFUSE_ENABLED": "1",
        "LANGFUSE_BASE_URL": "http://127.0.0.1:3100",
        "LANGFUSE_PROJECT_ID": "contract-analyzer",
        "LANGFUSE_PUBLIC_KEY": "pk-lf-" + secrets.token_hex(16),
        "LANGFUSE_SECRET_KEY": "sk-lf-" + secrets.token_hex(32),
        "LANGFUSE_INIT_USER_EMAIL": "reviewer@localhost.test",
        "LANGFUSE_INIT_USER_PASSWORD": secrets.token_urlsafe(24),
        **{key: secrets.token_hex(32) for key in (
            "POSTGRES_PASSWORD", "CLICKHOUSE_PASSWORD", "REDIS_AUTH", "MINIO_PASSWORD",
            "NEXTAUTH_SECRET", "SALT", "ENCRYPTION_KEY",
        )},
    }
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write("".join(f"{key}={value}\n" for key, value in values.items()))


if __name__ == "__main__":
    initialize(ROOT / ".env.langfuse")
    os.chdir(ROOT)
    print("Langfuse: http://127.0.0.1:3100; login credentials: .env.langfuse", flush=True)
    os.execvp("docker", ["docker", "compose", "--env-file", ".env.langfuse",
                        "-f", "compose.langfuse.yml", "up"])
