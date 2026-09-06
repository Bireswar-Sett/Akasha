from __future__ import annotations

import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests


def download_image_reference(
    url: str,
    max_bytes: int = 512 * 1024 * 1024,
) -> str:
    """
    Download a short-lived HTTPS signed URL into a private temporary file.

    The URL is deliberately never logged.
    """
    parsed = urlparse(url)

    if parsed.scheme != "https":
        raise ValueError("Only HTTPS signed URLs are accepted.")

    suffix = Path(parsed.path).suffix.lower() or ".tif"

    response = requests.get(
        url,
        stream=True,
        timeout=(15, 180),
        allow_redirects=False,
    )
    response.raise_for_status()

    directory = Path(
        tempfile.mkdtemp(prefix="akasha_image_")
    )
    destination = directory / f"input{suffix}"

    total = 0
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue

            total += len(chunk)
            if total > max_bytes:
                destination.unlink(missing_ok=True)
                raise ValueError("Downloaded image exceeds size limit.")

            handle.write(chunk)

    return str(destination)
