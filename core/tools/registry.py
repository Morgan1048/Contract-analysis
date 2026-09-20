"""The single tool registry: the chat agent's toolset today, and (task 08) the MCP
server's schemas and dispatch tomorrow. Neither reimplements the other.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict          # JSON Schema
    handler: Callable[..., dict]


_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _validate(parameters: dict, arguments: dict) -> list[str]:
    """Minimal JSON Schema validation: required fields, types, enums, numeric bounds.

    Covers exactly what the five tool schemas here use; not a general JSON Schema
    implementation.
    """
    errors: list[str] = []
    if not isinstance(arguments, dict):
        return ["arguments must be a JSON object"]

    properties: dict = parameters.get("properties", {})
    for name in parameters.get("required", []):
        if name not in arguments or arguments[name] is None:
            errors.append(f"missing required argument: {name!r}")

    for name, value in arguments.items():
        spec = properties.get(name)
        if spec is None:
            errors.append(f"unknown argument: {name!r}")
            continue
        expected = spec.get("type")
        py_type = _JSON_TYPES.get(expected)
        if isinstance(value, bool) and expected in {"integer", "number"}:
            # bool is a subclass of int; a JSON boolean must not satisfy a numeric schema.
            errors.append(f"argument {name!r} must be of type {expected!r}")
        elif py_type is not None and not isinstance(value, py_type):
            errors.append(f"argument {name!r} must be of type {expected!r}")
        elif expected in {"integer", "number"}:
            if "minimum" in spec and value < spec["minimum"]:
                errors.append(f"argument {name!r} must be >= {spec['minimum']}")
            if "maximum" in spec and value > spec["maximum"]:
                errors.append(f"argument {name!r} must be <= {spec['maximum']}")
        if "enum" in spec and value not in spec["enum"]:
            errors.append(f"argument {name!r} must be one of {spec['enum']}")
    return errors


def _apply_defaults(parameters: dict, arguments: dict) -> dict:
    merged = dict(arguments)
    for name, spec in parameters.get("properties", {}).items():
        if name not in merged and "default" in spec:
            merged[name] = spec["default"]
    return merged


class ToolRegistry:
    """The single definition of the tools available to the chat agent and (via
    ``mcp_specs``/``call``) the MCP server."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[dict]:
        """OpenAI-style function-calling specs."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def mcp_specs(self) -> list[dict]:
        """MCP tool schemas."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.parameters,
            }
            for tool in self._tools.values()
        ]

    def call(self, name: str, arguments: dict | None) -> dict[str, Any]:
        """Dispatch a tool call. Never raises: unknown tools, invalid arguments, and
        handler exceptions all come back as a structured error dict."""
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"unknown tool: {name!r}", "chunk_id": None, "found": False}

        arguments = arguments or {}
        errors = _validate(tool.parameters, arguments)
        if errors:
            return {"error": "; ".join(errors), "chunk_id": None, "found": False}

        try:
            result = tool.handler(**_apply_defaults(tool.parameters, arguments))
        except Exception as exc:  # the agent loop must never see a raised exception
            return {"error": f"{type(exc).__name__}: {exc}", "chunk_id": None, "found": False}

        if not result:
            return {"chunk_id": None, "found": False}
        return result
