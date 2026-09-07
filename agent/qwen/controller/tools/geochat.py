from __future__ import annotations

from typing import Any


def call(
    service: Any,
    image_url: str,
    prompt: str,
    max_new_tokens: int = 256,
    *,
    operation: str = "single_image_analysis",
    bounding_box: dict[str, Any] | None = None,
) -> Any:
    """
    Call the configured GeoChat service.

    The service owns the actual inference contract. This adapter only
    normalizes the arguments supplied by the controller/executor.

    Parameters
    ----------
    service:
        GeoChat service/client exposing an ``analyze`` method.

    image_url:
        Authorized image URL or local file reference.

    prompt:
        User/task-specific instruction.

    max_new_tokens:
        Generation limit.

    operation:
        Observable operation label. Kept out of the model call unless the
        deployed service explicitly supports it.

    bounding_box:
        Optional normalized region metadata. This is retained for callers
        and logging layers, but is not forwarded blindly to GeoChat because
        the deployed API contract may not accept it.
    """

    if not isinstance(image_url, str) or not image_url.strip():
        raise ValueError("image_url must be a non-empty string")

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    if isinstance(max_new_tokens, bool) or not isinstance(
        max_new_tokens,
        int,
    ):
        raise ValueError("max_new_tokens must be an integer")

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be greater than zero")

    if bounding_box is not None and not isinstance(
        bounding_box,
        dict,
    ):
        raise ValueError("bounding_box must be an object")

    # Keep the adapter intentionally thin. The executor is responsible for:
    # - resolving observation IDs
    # - downloading authorized signed URLs when needed
    # - SAR pseudo-RGB preprocessing
    # - error normalization
    # - dependency handling
    #
    # GeoChat itself should only receive the actual image + prompt contract
    # that its deployed service supports.
    return service.analyze(
        image_url=image_url.strip(),
        prompt=prompt.strip(),
        max_new_tokens=max_new_tokens,
    )