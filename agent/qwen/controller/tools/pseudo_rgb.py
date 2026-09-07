from __future__ import annotations

from typing import Any


def convert(
    image_url: str,
    crop: dict[str, int] | None = None,
) -> Any:
    """
    Convert a SAR source into the pseudo-RGB representation expected by
    the downstream visual model.

    Remote HTTPS references are downloaded first. Local filesystem paths
    are passed directly to the deterministic SAR processor.

    The actual pixel transformation is implemented by
    ``sar_to_pseudo_rgb_file``.
    """

    if not isinstance(image_url, str) or not image_url.strip():
        raise ValueError(
            "image_url must be a non-empty string"
        )

    if crop is not None and not isinstance(crop, dict):
        raise ValueError(
            "crop must be a dictionary when provided"
        )

    from qwen.controller.downloads import download_image_reference
    from qwen.controller.image_processing import (
        sar_to_pseudo_rgb_file,
    )

    image_url = image_url.strip()

    local_path = (
        download_image_reference(image_url)
        if image_url.startswith("https://")
        else image_url
    )

    return sar_to_pseudo_rgb_file(
        local_path,
        crop=crop,
    )