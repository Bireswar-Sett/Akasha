from __future__ import annotations

from typing import Any


def call(service: Any, image_1_url: str, image_2_url: str, prompt: str) -> Any:
    return service.analyze(image_1_url=image_1_url, image_2_url=image_2_url, prompt=prompt)
