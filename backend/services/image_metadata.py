from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any


def normalize_storage_metadata(image_path: str, blob: Any) -> dict[str, Any]:
    """Normalize only metadata supplied by Firebase Storage; unknown stays null."""
    custom = blob.metadata if isinstance(getattr(blob, "metadata", None), dict) else {}
    content_type = getattr(blob, "content_type", None) or custom.get("content_type")
    filename = PurePosixPath(image_path).name
    return {
        "filename": filename,
        "format": content_type or PurePosixPath(filename).suffix.lstrip(".").upper() or None,
        "modality": custom.get("modality"),
        "acquisition_time": custom.get("acquisition_time") or custom.get("timestamp"),
        "bands": custom.get("bands"),
        "polarization": custom.get("polarization"),
        "width": custom.get("width"),
        "height": custom.get("height"),
        "crs": custom.get("crs"),
        "georeferenced": custom.get("georeferenced"),
        "metadata_available": bool(custom or content_type),
    }
