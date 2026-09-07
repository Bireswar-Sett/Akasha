from __future__ import annotations

from typing import Any


def call(
    service: Any,
    image_t1_url: str,
    image_t2_url: str,
) -> Any:
    """
    Call the deployed M²CD change-detection service.

    The executor is responsible for resolving logical observations and
    preparing the actual image inputs. This adapter only forwards the
    two temporal image references using the service's public contract.
    """

    if not isinstance(image_t1_url, str) or not image_t1_url.strip():
        raise ValueError(
            "image_t1_url must be a non-empty string"
        )

    if not isinstance(image_t2_url, str) or not image_t2_url.strip():
        raise ValueError(
            "image_t2_url must be a non-empty string"
        )

    return service.detect_change(
        image_t1_url=image_t1_url.strip(),
        image_t2_url=image_t2_url.strip(),
    )