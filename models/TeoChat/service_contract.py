from __future__ import annotations


def require_temporal_pair(image_t1: str | None, image_t2: str | None) -> tuple[str, str]:
    if not image_t1 or not image_t2:
        raise ValueError("Exactly two optical images are required: T1 and T2.")
    return image_t1, image_t2
