#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
if [[ "${1:-}" == "--langfuse" ]]; then
  exec python3 scripts/start_langfuse.py
fi
if [[ "${1:-}" != "" && "${1:-}" != "--install-only" ]]; then
  echo "Usage: ./setup.sh [--install-only|--langfuse]" >&2
  exit 2
fi
if [[ ! -x .venv/bin/python ]]; then
  python3.12 -m venv .venv
fi
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -e ".[dev]"
./.venv/bin/python -c 'import tiktoken; tiktoken.get_encoding("o200k_harmony"); tiktoken.get_encoding("cl100k_base")'

echo "Setup complete. See docs/setup.md for Ollama, LM Studio, or OpenAI backend configuration."
if [[ "${1:-}" != "--install-only" ]]; then
  exec ./.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
fi
