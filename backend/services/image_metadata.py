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
        "role": custom.get("role") or custom.get("file_role"),
        # These fields are passed through only when a trusted upload/storage
        # workflow supplied them.  They are never inferred from filenames.
        "observation_id": custom.get("observation_id"),
        "relationship_type": custom.get("relationship_type"),
        "spatially_corresponding": _trusted_bool(custom.get("spatially_corresponding")),
        "co_registered": _trusted_bool(custom.get("co_registered")),
        "same_geographic_area": _trusted_bool(custom.get("same_geographic_area")),
        "width": custom.get("width"),
        "height": custom.get("height"),
        "size": getattr(blob, "size", None),
        "crs": custom.get("crs"),
        "georeferenced": custom.get("georeferenced"),
        "metadata_available": bool(custom or content_type),
    }


def _trusted_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None
