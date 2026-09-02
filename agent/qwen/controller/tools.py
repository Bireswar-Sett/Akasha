from __future__ import annotations

from typing import Any
import copy


# ======================================================================
# GeoChat
# ======================================================================

GEOCHAT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "geochat",
        "description": (
            "Analyze a single prepared remote-sensing image. "
            "Use this for one optical image or one already-prepared "
            "SAR pseudo-RGB image. Do not use this tool for temporal "
            "change detection between two observations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_url": {
                    "type": "string",
                    "description": (
                        "URL of the image that should be analyzed. "
                        "This may be a temporary signed URL."
                    ),
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "Specific visual-analysis instruction."
                    ),
                },
                "max_new_tokens": {
                    "type": "integer",
                    "description": (
                        "Maximum number of tokens GeoChat may generate."
                    ),
                    "minimum": 1,
                    "maximum": 512,
                    "default": 128,
                },
            },
            "required": [
                "image_url",
                "prompt",
            ],
            "additionalProperties": False,
        },
    },
}


# ======================================================================
# Registry
# ======================================================================

TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "geochat": GEOCHAT_TOOL,
}


# ======================================================================
# Helpers
# ======================================================================

def get_tool(
    name: str,
) -> dict[str, Any]:
    """
    Return a copy of a registered tool definition.
    """

    if name not in TOOL_REGISTRY:
        raise ValueError(
            f"Unknown tool: {name!r}"
        )

    return copy.deepcopy(
        TOOL_REGISTRY[name]
    )


def get_all_tools() -> list[dict[str, Any]]:
    """
    Return all registered tools.
    """

    return [
        copy.deepcopy(tool)
        for tool in TOOL_REGISTRY.values()
    ]