"""Thin MCP stdio transport over the shared core tool registry.

Business and tool behavior belongs in ``core/tools``. This module only translates
MCP JSON-RPC messages to registry calls and never defines a tool handler.
"""
from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from core.tools import ToolRegistry
from core.tools.contract_tools import build_mcp_registry


def dispatch(message: dict[str, Any], registry: ToolRegistry) -> dict[str, Any] | None:
    method = message.get("method")
    request_id = message.get("id")
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": request_id, "result": {
            "protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
            "serverInfo": {"name": "contract-analyzer", "version": "0.1.0"},
        }}
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": registry.mcp_specs()}}
    if method == "tools/call":
        params = message.get("params") or {}
        name = params.get("name")
        result = registry.call(name, params.get("arguments"))
        is_error = "error" in result
        return {"jsonrpc": "2.0", "id": request_id, "result": {"isError": is_error, "structuredContent": result, "content": [{"type": "text", "text": json.dumps(result)}]}}
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": f"method not found: {method!r}"}}


def serve_stdio(registry: ToolRegistry, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
    """Return per-frame protocol errors without terminating the transport."""
    for line in stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            stdout.write(json.dumps({
                "jsonrpc": "2.0", "id": None,
                "error": {"code": -32700, "message": "parse error"},
            }) + "\n")
            stdout.flush()
            continue
        try:
            response = dispatch(message, registry)
        except Exception:
            response = {
                "jsonrpc": "2.0", "id": message.get("id") if isinstance(message, dict) else None,
                "error": {"code": -32603, "message": "internal error"},
            }
        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()


def main() -> None:
    serve_stdio(build_mcp_registry())


if __name__ == "__main__":
    main()
