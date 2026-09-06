from __future__ import annotations

import copy
from typing import Any


def _function_tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False}}}


TOOL_REGISTRY = {
    "geochat": _function_tool("geochat", "Analyze one prepared remote-sensing image.", {"image_ref": {"type": "string"}, "prompt": {"type": "string"}, "max_new_tokens": {"type": "integer", "minimum": 32, "maximum": 512}}, ["image_ref", "prompt"]),
    "teochat": _function_tool("teochat", "Analyze a supported paired optical and SAR request.", {"image_1_ref": {"type": "string"}, "image_2_ref": {"type": "string"}, "prompt": {"type": "string"}}, ["image_1_ref", "image_2_ref", "prompt"]),
    "m2cd": _function_tool("m2cd", "Detect spatial change between corresponding observations.", {"image_t1_ref": {"type": "string"}, "image_t2_ref": {"type": "string"}}, ["image_t1_ref", "image_t2_ref"]),
    "pseudo_rgb": _function_tool("pseudo_rgb", "Convert VV/VH SAR into robust pseudo-RGB.", {"image_ref": {"type": "string"}, "crop": {"type": "object"}}, ["image_ref"]),
}


def get_tool(name: str) -> dict[str, Any]:
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {name!r}")
    return copy.deepcopy(TOOL_REGISTRY[name])


def get_all_tools() -> list[dict[str, Any]]:
    return [copy.deepcopy(tool) for tool in TOOL_REGISTRY.values()]
