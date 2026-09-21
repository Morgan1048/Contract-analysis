from core.tools.registry import Tool, ToolRegistry


def build_tool_registry(*args, **kwargs):
    from core.tools.contract_tools import build_tool_registry as build
    return build(*args, **kwargs)

__all__ = ["Tool", "ToolRegistry", "build_tool_registry"]
