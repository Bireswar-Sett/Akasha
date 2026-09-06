from __future__ import annotations

import re
from dataclasses import dataclass

from qwen.controller.schemas import BoundingBox


class BoundingBoxError(ValueError):
    pass


_BOX_TOKEN = re.compile(
    r"b\s*=\s*\{\s*"
    r"(?P<x_left>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"(?P<y_top>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"(?P<x_right>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"(?P<y_bottom>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*\|\s*"
    r"(?P<angle>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*\}",
)


@dataclass(frozen=True)
class ParsedQuery:
    user_query: str
    bounding_boxes: list[BoundingBox]


def parse_query(value: str) -> ParsedQuery:
    query = value.strip()
    matches = list(_BOX_TOKEN.finditer(query))
    marker = re.search(r"(?:^|\s)b\s*=", query)
    if not matches:
        if marker:
            raise BoundingBoxError("malformed bounding-box syntax")
        return ParsedQuery(user_query=query, bounding_boxes=[])

    suffix = query[matches[0].start():]
    expected_suffix = "".join(match.group(0) for match in matches)
    if suffix.replace(" ", "").replace("\n", "") != expected_suffix.replace(" ", ""):
        raise BoundingBoxError("bounding-box specifications must appear at the end of the query")

    boxes: list[BoundingBox] = []
    for match in matches:
        try:
            boxes.append(BoundingBox(**{key: float(match.group(key)) for key in ("x_left", "y_top", "x_right", "y_bottom", "angle")}))
        except ValueError as exc:
            raise BoundingBoxError(str(exc)) from exc

    user_query = query[:matches[0].start()].strip()
    if not user_query:
        raise BoundingBoxError("a bounding box must accompany a user request")
    return ParsedQuery(user_query=user_query, bounding_boxes=boxes)
