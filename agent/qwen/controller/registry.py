from __future__ import annotations

from typing import Any

from qwen.controller.tools import get_all_tools


TOOL_CAPABILITIES: dict[str, set[str]] = {
    "geochat": {"single_image", "semantic_interpretation", "region_grounding"},
    "teochat": {"cross_modal_reasoning"},
    "m2cd": {"change_detection", "change_localization"},
    "pseudo_rgb": {"sar_representation"},
}


def tool_registry() -> list[dict[str, Any]]:
    """Return immutable tool schemas; this module never executes tools."""
    return get_all_tools()


def supports(tool_name: str, capability: str) -> bool:
    return capability in TOOL_CAPABILITIES.get(tool_name, set())
