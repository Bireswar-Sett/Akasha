from __future__ import annotations

from typing import Any


def convert(image_url: str, crop: dict[str, int] | None = None) -> Any:
    from qwen.controller.image_processing import sar_to_pseudo_rgb_file
    from qwen.controller.downloads import download_image_reference

    local_path = download_image_reference(image_url) if image_url.startswith("https://") else image_url
    return sar_to_pseudo_rgb_file(local_path, crop=crop)
