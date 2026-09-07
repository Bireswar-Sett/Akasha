"""
Normalization of trusted Firebase Storage metadata.

This module is deliberately conservative:

- Firebase Storage metadata is the source of truth.
- No modality, polarization, timestamp, or observation relationship is
  inferred from filenames.
- Missing metadata remains None.
- Values are normalized into the schema expected by input_manifest.py.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any


def normalize_storage_metadata(
    image_path: str,
    blob: Any,
) -> dict[str, Any]:
    """
    Normalize trusted Firebase Storage metadata into a stable internal form.

    The function does not inspect the filename to infer scientific metadata.
    The filename is retained only as an identifier/display field.
    """

    custom = (
        blob.metadata
        if isinstance(
            getattr(blob, "metadata", None),
            dict,
        )
        else {}
    )

    content_type = (
        getattr(blob, "content_type", None)
        or custom.get("content_type")
    )

    filename = PurePosixPath(
        image_path
    ).name

    extension = (
        PurePosixPath(filename)
        .suffix
        .lstrip(".")
        .lower()
        or None
    )

    normalized_format = _normalize_format(
        custom.get("file_format")
        or custom.get("format")
        or content_type
        or extension
    )

    return {
        # ---------------------------------------------------------------
        # File identity
        # ---------------------------------------------------------------

        "filename": (
            custom.get("original_filename")
            or filename
        ),

        "format": normalized_format,

        "size": getattr(
            blob,
            "size",
            None,
        ),

        # ---------------------------------------------------------------
        # Remote-sensing metadata
        #
        # NEVER infer these from filename here.
        # ---------------------------------------------------------------

        "modality": _normalize_modality(
            custom.get("modality")
        ),

        "acquisition_time": _normalize_string(
            custom.get("acquisition_time")
            or custom.get("timestamp")
        ),

        "bands": _normalize_optional_value(
            custom.get("bands")
        ),

        "polarization": _normalize_polarization(
            custom.get("polarization")
            or custom.get("role")
            or custom.get("file_role")
        ),

        # ---------------------------------------------------------------
        # Logical observation grouping
        # ---------------------------------------------------------------

        "observation_id": _normalize_string(
            custom.get("observation_id")
        ),

        "role": _normalize_string(
            custom.get("role")
            or custom.get("file_role")
            or custom.get("imagery_role")
        ),

        # ---------------------------------------------------------------
        # Relationship metadata
        # ---------------------------------------------------------------

        "relationship_type": _normalize_relationship(
            custom.get("relationship_type")
        ),

        "spatially_corresponding": _trusted_bool(
            custom.get("spatially_corresponding")
        ),

        "co_registered": _trusted_bool(
            custom.get("co_registered")
        ),

        "same_geographic_area": _trusted_bool(
            custom.get("same_geographic_area")
        ),

        # ---------------------------------------------------------------
        # Raster metadata
        # ---------------------------------------------------------------

        "width": _normalize_optional_value(
            custom.get("width")
        ),

        "height": _normalize_optional_value(
            custom.get("height")
        ),

        "crs": _normalize_string(
            custom.get("crs")
        ),

        "georeferenced": _trusted_bool(
            custom.get("georeferenced")
        ),

        "geocoded": _trusted_bool(
            custom.get("geocoded")
        ),

        # ---------------------------------------------------------------
        # Benchmark / provenance metadata
        # ---------------------------------------------------------------

        "benchmark_dataset": _normalize_string(
            custom.get("benchmark_dataset")
        ),

        "imagery_role": _normalize_string(
            custom.get("imagery_role")
        ),

        "metadata_version": _normalize_string(
            custom.get("metadata_version")
        ),

        # ---------------------------------------------------------------
        # Availability indicator
        # ---------------------------------------------------------------

        "metadata_available": bool(
            custom or content_type
        ),
    }


# ---------------------------------------------------------------------------
# Generic normalization
# ---------------------------------------------------------------------------


def _normalize_string(
    value: Any,
) -> str | None:
    if value is None:
        return None

    if not isinstance(value, str):
        return str(value)

    normalized = value.strip()

    return normalized or None


def _normalize_optional_value(
    value: Any,
) -> Any:
    if value is None:
        return None

    if isinstance(value, str):
        normalized = value.strip()

        if not normalized:
            return None

        return normalized

    return value


# ---------------------------------------------------------------------------
# File format
# ---------------------------------------------------------------------------


def _normalize_format(
    value: Any,
) -> str | None:
    """
    Normalize formats such as:

        image/tiff
        image/geotiff
        TIFF
        .tif
        .tiff

    into canonical values:

        tif
        tiff
        png
        jpg
        jpeg

    Do not attempt to infer anything beyond the supplied format value.
    """

    if not isinstance(value, str):
        return None

    value = value.strip().lower()

    if not value:
        return None

    # MIME type → extension
    mime_map = {
        "image/tiff": "tiff",
        "image/geotiff": "tiff",
        "image/png": "png",
        "image/jpeg": "jpeg",
        "image/jpg": "jpg",
    }

    if value in mime_map:
        return mime_map[value]

    # Remove a leading dot.
    value = value.lstrip(".")

    if value == "geotiff":
        return "tiff"

    return value


# ---------------------------------------------------------------------------
# Modality
# ---------------------------------------------------------------------------


def _normalize_modality(
    value: Any,
) -> str | None:
    if not isinstance(value, str):
        return None

    value = value.strip().lower()

    if value in {
        "optical",
        "multispectral",
        "sar",
    }:
        return value

    return None


# ---------------------------------------------------------------------------
# Polarization
# ---------------------------------------------------------------------------


def _normalize_polarization(
    value: Any,
) -> str | None:
    if not isinstance(value, str):
        return None

    value = value.strip().upper()

    # Accept metadata values such as:
    #
    #   SAR_VV
    #   SAR_VH
    #   VV
    #   VH
    #
    if value.startswith("SAR_"):
        value = value[4:]

    if value in {
        "VV",
        "VH",
    }:
        return value

    return None


# ---------------------------------------------------------------------------
# Relationship
# ---------------------------------------------------------------------------


def _normalize_relationship(
    value: Any,
) -> str | None:
    if not isinstance(value, str):
        return None

    value = value.strip().lower()

    aliases = {
        "temporal": "temporal",
        "bi_temporal": "bi_temporal",
        "bitemporal": "bi_temporal",
        "cross_modal": "cross_modal",
        "crossmodal": "cross_modal",
        "single": "single",
    }

    return aliases.get(value)


# ---------------------------------------------------------------------------
# Trusted booleans
# ---------------------------------------------------------------------------


def _trusted_bool(
    value: Any,
) -> bool | None:
    """
    Convert only explicit trusted boolean representations.

    Never use truthiness such as bool(value), because:

        bool("false") == True

    which is exactly the sort of tiny bug that ruins an afternoon.
    """

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized == "true":
            return True

        if normalized == "false":
            return False

    return None