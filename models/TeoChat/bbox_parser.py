from __future__ import annotations

import re
from typing import Any


_BOX_PATTERN = re.compile(
    r"\[\s*(?P<x_left>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y_top>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<x_right>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y_bottom>-?\d+(?:\.\d+)?)\s*\]"
)
_CANONICAL_PATTERN = re.compile(
    r"\{\s*(?P<x_left>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y_top>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<x_right>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y_bottom>-?\d+(?:\.\d+)?)\s*\|\s*"
    r"(?P<angle>-?\d+(?:\.\d+)?)\s*\}"
)
_IMAGE_MARKER = re.compile(r"(?:image|frame)\s*(?P<index>\d+)", re.IGNORECASE)


class BoundingBoxError(ValueError):
    pass


def _box(values: dict[str, str], angle: float = 0.0) -> dict[str, float]:
    result = {key: float(values[key]) for key in ("x_left", "y_top", "x_right", "y_bottom")}
    result["angle"] = angle
    if any(value < 0 or value > 100 for key, value in result.items() if key != "angle"):
        raise BoundingBoxError("TEOChat bounding-box coordinates must be normalized to [0, 100]")
    if result["x_left"] > result["x_right"] or result["y_top"] > result["y_bottom"]:
        raise BoundingBoxError("TEOChat bounding-box geometry is invalid")
    return result


def _image_index(text: str, start: int, image_count: int) -> int:
    prefix = text[max(0, start - 80):start]
    markers = list(_IMAGE_MARKER.finditer(prefix))
    index = int(markers[-1].group("index")) - 1 if markers else 0
    if not 0 <= index < image_count:
        raise BoundingBoxError("TEOChat bbox image index is out of range")
    return index


def extract_bboxes(text: str) -> list[list[float]]:
    """Backward-compatible extraction of the repository's [x1,y1,x2,y2] format."""
    return [
        [int(match.group(name)) for name in ("x_left", "y_top", "x_right", "y_bottom")]
        for match in _BOX_PATTERN.finditer(text or "")
    ]


def normalize_bboxes(text: str, image_count: int) -> list[dict[str, Any]]:
    """Convert model bbox text into image-associated canonical evidence."""
    if image_count < 1:
        raise ValueError("image_count must be positive")
    output: list[dict[str, Any]] = []
    canonical_matches = list(_CANONICAL_PATTERN.finditer(text or ""))
    occupied = [(match.start(), match.end()) for match in canonical_matches]
    matches: list[tuple[int, Any, bool]] = [(match.start(), match, True) for match in canonical_matches]
    for match in _BOX_PATTERN.finditer(text or ""):
        if any(start < match.end() and match.start() < end for start, end in occupied):
            continue
        matches.append((match.start(), match, False))
    for _, match, canonical in sorted(matches, key=lambda item: item[0]):
        image_index = _image_index(text, match.start(), image_count)
        angle = float(match.group("angle")) if canonical else 0.0
        output.append({"image_index": image_index, "box": _box(match.groupdict(), angle)})
    return output
