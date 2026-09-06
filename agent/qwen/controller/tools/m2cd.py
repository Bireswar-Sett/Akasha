from __future__ import annotations

from typing import Any


def call(service: Any, image_t1_url: str, image_t2_url: str) -> Any:
    return service.detect_change(image_t1_url=image_t1_url, image_t2_url=image_t2_url)
