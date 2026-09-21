from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

_SECRET = re.compile(r"((?:sk-|gsk_)[A-Za-z0-9_-]+|(?:Bearer|Basic)\s+[A-Za-z0-9._=+/-]+|api[_-]?key[=:]\s*\S+)", re.I)
_DEFAULT_LOG_MAX_BYTES = 10 * 1024 * 1024
_DEFAULT_LOG_BACKUPS = 5


class PrivateRotatingFileHandler(RotatingFileHandler):
    def _open(self):
        fd = os.open(self.baseFilename, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.fchmod(fd, 0o600)
        return os.fdopen(fd, self.mode, encoding=self.encoding, errors=self.errors)


def _positive_int_env(name: str, default: int) -> int:
    """Read a positive logging limit without turning bad configuration into no limit."""
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


class JsonFormatter(logging.Formatter):
    def format(self, record):
        fields = getattr(record, "fields", {}).copy()
        fields.update(ts=record.created, level=record.levelname.lower(), event=record.msg)
        def scrub(value):
            if isinstance(value, str):
                return _SECRET.sub("[REDACTED]", value)
            if isinstance(value, dict):
                return {k: "[REDACTED]" if re.search(r"(^|_)(api_key|secret_key|password|authorization)$", k, re.I)
                        else scrub(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [scrub(v) for v in value]
            return value
        return json.dumps({k: scrub(v) for k, v in fields.items() if v is not None}, default=str)


def get_logger(name="contract-analyzer", *, stream=None, enable_stream: bool | None = None):
    """Build (or fetch) the process logger.

    Never attaches a handler to ``sys.stdout``: task 08 serves the MCP server over
    stdio, where JSON-RPC frames travel on stdout, and interleaved log lines would
    corrupt that protocol. The stream handler defaults to ``stderr`` and can be
    disabled entirely (e.g. by an MCP server process) via ``enable_stream=False`` or
    the ``CA_LOG_STREAM`` environment variable.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = JsonFormatter()
        if enable_stream is None:
            enable_stream = os.getenv("CA_LOG_STREAM", "1").strip().lower() not in {"0", "false", "no"}
        if enable_stream:
            handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        path = Path("logs/contract-analyzer.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = PrivateRotatingFileHandler(
            path,
            maxBytes=_positive_int_env("CA_LOG_MAX_BYTES", _DEFAULT_LOG_MAX_BYTES),
            backupCount=_positive_int_env("CA_LOG_BACKUP_COUNT", _DEFAULT_LOG_BACKUPS),
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.propagate = False
    return logger
