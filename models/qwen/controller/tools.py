from __future__ import annotations

from typing import Any


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
                "image_path": {
                    "type": "string",
                    "description": (
                        "Local path to the image that should be "
                        "sent to the GeoChat service."
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
                "image_path",
                "prompt",
            ],
            "additionalProperties": False,
        },
    },
}


# ======================================================================
# TeoChat
# ======================================================================

TEOCHAT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "teochat",
        "description": (
            "Compare two optical remote-sensing observations "
            "and describe meaningful changes between them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "before_image": {
                    "type": "string",
                    "description": (
                        "Local path to the earlier optical image."
                    ),
                },
                "after_image": {
                    "type": "string",
                    "description": (
                        "Local path to the later optical image."
                    ),
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "Instruction describing what temporal "
                        "changes should be examined."
                    ),
                },
                "max_new_tokens": {
                    "type": "integer",
                    "description": (
                        "Maximum number of tokens TeoChat may generate."
                    ),
                    "minimum": 1,
                    "maximum": 512,
                    "default": 128,
                },
            },
            "required": [
                "before_image",
                "after_image",
                "prompt",
            ],
            "additionalProperties": False,
        },
    },
}


# ======================================================================
# M2CD
# ======================================================================

M2CD_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "m2cd",
        "description": (
            "Run SAR change detection between two Sentinel-1 "
            "observations. Use this when two SAR observations "
            "need to be compared. Returns a change-probability "
            "mask and related metadata."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "before_vv": {
                    "type": "string",
                    "description": (
                        "Local path to earlier VV raster."
                    ),
                },
                "before_vh": {
                    "type": "string",
                    "description": (
                        "Local path to earlier VH raster."
                    ),
                },
                "after_vv": {
                    "type": "string",
                    "description": (
                        "Local path to later VV raster."
                    ),
                },
                "after_vh": {
                    "type": "string",
                    "description": (
                        "Local path to later VH raster."
                    ),
                },
            },
            "required": [
                "before_vv",
                "before_vh",
                "after_vv",
                "after_vh",
            ],
            "additionalProperties": False,
        },
    },
}


# ======================================================================
# SAR RGB
# ======================================================================

MAKE_SAR_RGB_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "make_sar_rgb",
        "description": (
            "Construct the GeoChat-compatible pseudo-RGB image "
            "from one Sentinel-1 VV/VH pair. The deterministic "
            "channel mapping is: R = VV, G = VH, "
            "B = (VV + VH) / 2."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "vv_path": {
                    "type": "string",
                    "description": (
                        "Local path to the VV raster."
                    ),
                },
                "vh_path": {
                    "type": "string",
                    "description": (
                        "Local path to the VH raster."
                    ),
                },
                "output_path": {
                    "type": "string",
                    "description": (
                        "Optional local output path for the generated "
                        "RGB image."
                    ),
                },
            },
            "required": [
                "vv_path",
                "vh_path",
            ],
            "additionalProperties": False,
        },
    },
}


# ======================================================================
# Changed-region cropper
# ======================================================================

CROP_CHANGED_REGION_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "crop_changed_region",
        "description": (
            "Use a change-probability mask to retain the changed "
            "pixels from one or more source rasters. A pixel is "
            "selected when its change probability is greater than "
            "the supplied threshold."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mask_path": {
                    "type": "string",
                    "description": (
                        "Local path to the M2CD change-probability mask."
                    ),
                },
                "input_paths": {
                    "type": "array",
                    "description": (
                        "Local paths to the source rasters that "
                        "should be filtered using the mask."
                    ),
                    "items": {
                        "type": "string",
                    },
                },
                "threshold": {
                    "type": "number",
                    "description": (
                        "Minimum probability required for a pixel "
                        "to be retained."
                    ),
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.5,
                },
            },
            "required": [
                "mask_path",
                "input_paths",
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
    "teochat": TEOCHAT_TOOL,
    "m2cd": M2CD_TOOL,
    "make_sar_rgb": MAKE_SAR_RGB_TOOL,
    "crop_changed_region": CROP_CHANGED_REGION_TOOL,
}


def get_tool(
    name: str,
) -> dict[str, Any]:
    """
    Return a tool definition.
    """

    if name not in TOOL_REGISTRY:
        raise ValueError(
            f"Unknown tool: {name!r}"
        )

    # Avoid exposing/mutating the global registry object.
    import copy

    return copy.deepcopy(
        TOOL_REGISTRY[name]
    )


def get_all_tools() -> list[dict[str, Any]]:
    """
    Return all registered tool definitions.
    """

    import copy

    return [
        copy.deepcopy(tool)
        for tool in TOOL_REGISTRY.values()
    ]