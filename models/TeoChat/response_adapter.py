from __future__ import annotations

from typing import Any

try:
    from bbox_parser import normalize_bboxes
except ModuleNotFoundError:
    from models.TeoChat.bbox_parser import normalize_bboxes


def normalize_response(answer: str, image_count: int, model: str = "TEOChat") -> dict[str, Any]:
    """Return JSON-compatible specialist evidence without inventing confidence."""
    bbox_error = None
    try:
        boxes = normalize_bboxes(answer, image_count)
    except ValueError as exc:
        boxes = []
        bbox_error = str(exc)
    return {
        "status": "completed",
        "tool": model,
        "answer": answer,
        "bounding_boxes": boxes,
        "input_regions": [],
        "evidence": {
            "observations": [answer] if answer else [],
            "interpretations": [],
            "spatial_outputs": boxes,
            "confidence": None,
            "uncertainty": bbox_error,
        },
    }
