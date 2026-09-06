from __future__ import annotations

from typing import Any


def call(service: Any, image_url: str, prompt: str, max_new_tokens: int = 256) -> Any:
    return service.analyze(image_url=image_url, prompt=prompt, max_new_tokens=max_new_tokens)
