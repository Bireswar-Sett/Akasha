"""
Trusted backend construction of Qwen-compatible input manifests.

This module converts physical S3 object metadata into logical
remote-sensing observations.

Important:
- Modality, polarization, timestamps, and observation IDs come from
  trusted S3-verified file metadata supplied by the route layer.
- This module NEVER infers modality/polarization from filenames.
- A SAR observation consists of exactly one VV file + one VH file.
- The application supports at most four physical files.
- Relationship information belongs to the logical observations, not
  necessarily to every physical file.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any


class InputManifestCompatibilityError(ValueError):
    """Raised when Storage metadata cannot safely describe the uploads."""


_SAFE_ERROR = (
    "The uploaded files cannot be safely grouped into supported observations. "
    "Please upload a supported imagery combination."
)

_MAX_PHYSICAL_FILES = 4

_SUPPORTED_MODALITIES = {
    "optical",
    "multispectral",
    "sar",
}

_SUPPORTED_POLARIZATIONS = {
    "VV",
    "VH",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_input_manifest(
    file_metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Convert trusted physical-file metadata into a logical input manifest.

    Supported physical-file configurations:

        1 file
            optical / multispectral

        2 files
            optical + optical
            SAR VV + VH

        3 files
            optical + SAR VV + SAR VH

        4 files
            SAR(T1) VV + SAR(T1) VH
            SAR(T2) VV + SAR(T2) VH

    The resulting manifest explicitly separates:

        physical_files
        observations
        relationship
    """

    if not isinstance(file_metadata, list):
        raise InputManifestCompatibilityError(_SAFE_ERROR)

    if not 1 <= len(file_metadata) <= _MAX_PHYSICAL_FILES:
        raise InputManifestCompatibilityError(
            "between one and four image files are supported"
        )

    if any(not isinstance(item, dict) for item in file_metadata):
        raise InputManifestCompatibilityError(_SAFE_ERROR)

    files = [
        _physical_file(index, metadata)
        for index, metadata in enumerate(file_metadata)
    ]

    modalities = [
        _normal_modality(metadata.get("modality"))
        for metadata in file_metadata
    ]

    if any(modality is None for modality in modalities):
        raise InputManifestCompatibilityError(_SAFE_ERROR)

    # -----------------------------------------------------------------------
    # Single physical file
    # -----------------------------------------------------------------------

    if len(file_metadata) == 1:
        modality = modalities[0]

        if modality not in {"optical", "multispectral"}:
            raise InputManifestCompatibilityError(_SAFE_ERROR)

        observation = _image_observation(
            observation_id="observation_1",
            modality=modality,
            file=files[0],
            metadata=file_metadata[0],
        )

        return _manifest(
            files=files,
            observations=[observation],
            relationship={"type": "single"},
        )

    sar_indices = [
        index
        for index, modality in enumerate(modalities)
        if modality == "sar"
    ]

    optical_indices = [
        index
        for index, modality in enumerate(modalities)
        if modality in {"optical", "multispectral"}
    ]

    # -----------------------------------------------------------------------
    # Two optical observations
    # -----------------------------------------------------------------------

    if len(file_metadata) == 2 and len(optical_indices) == 2:
        relationship = _temporal_relationship(file_metadata)

        observations = [
            _image_observation(
                observation_id=f"observation_{position + 1}",
                modality=modalities[index],
                file=files[index],
                metadata=file_metadata[index],
            )
            for position, index in enumerate(optical_indices)
        ]

        _validate_temporal_order(
            observations,
            file_metadata,
        )

        return _manifest(
            files=files,
            observations=observations,
            relationship=relationship,
        )

    # -----------------------------------------------------------------------
    # One SAR observation = VV + VH
    # -----------------------------------------------------------------------

    if len(file_metadata) == 2 and len(sar_indices) == 2:
        observation = _sar_observation(
            observation_id="observation_1",
            indices=sar_indices,
            files=files,
            metadata=file_metadata,
        )

        return _manifest(
            files=files,
            observations=[observation],
            relationship={"type": "single"},
        )

    # -----------------------------------------------------------------------
    # Optical + SAR
    #
    # Physical files:
    #     optical
    #     SAR VV
    #     SAR VH
    #
    # Logical observations:
    #     observation_optical
    #     observation_sar
    #
    # IMPORTANT:
    # Relationship metadata such as co-registration is NOT required
    # to be duplicated on all three physical files.
    # -----------------------------------------------------------------------

    if (
        len(file_metadata) == 3
        and len(optical_indices) == 1
        and len(sar_indices) == 2
    ):
        relationship = _cross_modal_relationship(
            file_metadata=file_metadata,
        )

        optical_index = optical_indices[0]

        observations = [
            _image_observation(
                observation_id="observation_optical",
                modality=modalities[optical_index],
                file=files[optical_index],
                metadata=file_metadata[optical_index],
            ),
            _sar_observation(
                observation_id="observation_sar",
                indices=sar_indices,
                files=files,
                metadata=file_metadata,
            ),
        ]

        return _manifest(
            files=files,
            observations=observations,
            relationship=relationship,
        )

    # -----------------------------------------------------------------------
    # Two SAR observations
    #
    # Physical files:
    #
    #     T1 VV
    #     T1 VH
    #     T2 VV
    #     T2 VH
    #
    # Logical observations:
    #
    #     observation_1 = SAR(T1)
    #     observation_2 = SAR(T2)
    #
    # Each pair MUST have its own observation_id.
    # -----------------------------------------------------------------------

    if len(file_metadata) == 4 and len(sar_indices) == 4:
        grouped = _sar_groups(
            sar_indices=sar_indices,
            metadata=file_metadata,
        )

        if len(grouped) != 2:
            raise InputManifestCompatibilityError(_SAFE_ERROR)

        ordered_groups = _order_sar_groups(
            grouped,
            file_metadata,
        )

        observations = [
            _sar_observation(
                observation_id=f"observation_{position + 1}",
                indices=group,
                files=files,
                metadata=file_metadata,
            )
            for position, group in enumerate(ordered_groups)
        ]

        _validate_temporal_observations(observations)

        relationship = _temporal_relationship(
            file_metadata=file_metadata,
        )

        return _manifest(
            files=files,
            observations=observations,
            relationship=relationship,
        )

    # -----------------------------------------------------------------------
    # Anything outside the supported physical configurations is rejected.
    # -----------------------------------------------------------------------

    raise InputManifestCompatibilityError(_SAFE_ERROR)


# ---------------------------------------------------------------------------
# Physical file representation
# ---------------------------------------------------------------------------


def _physical_file(
    index: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Normalize one physical file.

    No filename parsing occurs here.
    """

    keys = (
        "filename",
        "format",
        "size",
        "role",
        "modality",
        "acquisition_time",
        "bands",
        "polarization",
        "observation_id",
        "relationship_type",
        "spatially_corresponding",
        "co_registered",
        "same_geographic_area",
        "georeferenced",
        "geocoded",
        "benchmark_dataset",
    )

    result = {
        "id": f"file_{index}",
    }

    for key in keys:
        value = metadata.get(key)

        if value is not None:
            result[key] = value

    return result


# ---------------------------------------------------------------------------
# Optical / multispectral observation
# ---------------------------------------------------------------------------


def _image_observation(
    observation_id: str,
    modality: str,
    file: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    observation = {
        "id": observation_id,
        "modality": modality,
        "image": {
            "id": file["id"],
        },
    }

    acquisition_time = _normalize_timestamp(
        metadata.get("acquisition_time")
    )

    if acquisition_time is not None:
        observation["acquisition_time"] = acquisition_time

    if metadata.get("filename") is not None:
        observation["image"]["filename"] = metadata["filename"]

    return observation


# ---------------------------------------------------------------------------
# SAR observation
# ---------------------------------------------------------------------------


def _sar_observation(
    observation_id: str,
    indices: list[int],
    files: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build exactly one logical SAR observation from VV + VH.

    Requirements:
        - exactly two physical files
        - one VV
        - one VH
        - same acquisition time
    """

    if len(indices) != 2:
        raise InputManifestCompatibilityError(_SAFE_ERROR)

    polarization_to_index: dict[str, int] = {}

    for index in indices:
        polarization = _polarization(
            metadata[index].get("polarization")
            or metadata[index].get("role")
        )

        if polarization is None:
            raise InputManifestCompatibilityError(_SAFE_ERROR)

        if polarization in polarization_to_index:
            # Duplicate VV or VH.
            raise InputManifestCompatibilityError(_SAFE_ERROR)

        polarization_to_index[polarization] = index

    if set(polarization_to_index) != {"VV", "VH"}:
        raise InputManifestCompatibilityError(_SAFE_ERROR)

    timestamp_values = {
        _normalize_timestamp(
            metadata[index].get("acquisition_time")
        )
        for index in indices
    }

    if len(timestamp_values) != 1:
        raise InputManifestCompatibilityError(_SAFE_ERROR)

    acquisition_time = timestamp_values.pop()

    observation = {
        "id": observation_id,
        "modality": "sar",
        "sar": {
            "vv": {
                "id": files[
                    polarization_to_index["VV"]
                ]["id"],
            },
            "vh": {
                "id": files[
                    polarization_to_index["VH"]
                ]["id"],
            },
        },
    }

    if acquisition_time is not None:
        observation["acquisition_time"] = acquisition_time

    # Preserve format information where available.
    for polarization, key in (
        ("VV", "vv"),
        ("VH", "vh"),
    ):
        index = polarization_to_index[polarization]

        if metadata[index].get("format") is not None:
            observation["sar"][key]["format"] = (
                metadata[index]["format"]
            )

    return observation


# ---------------------------------------------------------------------------
# SAR grouping
# ---------------------------------------------------------------------------


def _sar_groups(
    sar_indices: list[int],
    metadata: list[dict[str, Any]],
) -> list[list[int]]:
    """
    Group physical SAR files into logical observations.

    Every four-file dual-SAR request must provide an explicit
    `observation_id` in trusted Storage metadata.

    The frontend assigns the same observation_id to VV/VH belonging
    to the same acquisition.
    """

    groups: dict[str, list[int]] = defaultdict(list)

    for index in sar_indices:
        observation_id = metadata[index].get(
            "observation_id"
        )

        if not isinstance(observation_id, str) or not observation_id.strip():
            raise InputManifestCompatibilityError(_SAFE_ERROR)

        normalized_id = observation_id.strip()
        groups[normalized_id].append(index)

    # Exactly two logical SAR observations are required.
    if len(groups) != 2:
        raise InputManifestCompatibilityError(_SAFE_ERROR)

    # Every logical SAR observation must have exactly two physical files.
    if any(
        len(indices) != 2
        for indices in groups.values()
    ):
        raise InputManifestCompatibilityError(_SAFE_ERROR)

    return list(groups.values())


def _order_sar_groups(
    groups: list[list[int]],
    metadata: list[dict[str, Any]],
) -> list[list[int]]:
    """
    Put SAR observations in acquisition-time order.

    If timestamps cannot be parsed into comparable datetime values,
    retain grouping order instead of guessing.
    """

    decorated = []

    for original_position, group in enumerate(groups):
        timestamps = {
            _normalize_timestamp(
                metadata[index].get("acquisition_time")
            )
            for index in group
        }

        if len(timestamps) != 1:
            raise InputManifestCompatibilityError(_SAFE_ERROR)

        timestamp = next(iter(timestamps))

        parsed = _parse_timestamp(timestamp)

        decorated.append(
            (
                parsed is None,
                parsed,
                original_position,
                group,
            )
        )

    # If every timestamp is parseable, chronological ordering is safe.
    if all(item[1] is not None for item in decorated):
        decorated.sort(
            key=lambda item: (
                item[1],
                item[2],
            )
        )

    return [
        item[3]
        for item in decorated
    ]


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------


def _temporal_relationship(
    file_metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata = file_metadata

    timestamps = [
        item.get("acquisition_time")
        for item in metadata
    ]

    if any(timestamp is None for timestamp in timestamps):
        raise InputManifestCompatibilityError(
            "Temporal inputs require acquisition_time metadata for every "
            "physical file."
        )

    relationship: dict[str, Any] = {
        "type": "temporal",
    }

    spatially_corresponding = _uniform_boolean(
        metadata,
        "spatially_corresponding",
    )

    if spatially_corresponding is not None:
        relationship["spatially_corresponding"] = spatially_corresponding

    same_geographic_area = _uniform_boolean(
        metadata,
        "same_geographic_area",
    )

    if same_geographic_area is not None:
        relationship["same_geographic_area"] = same_geographic_area

    return relationship


def _cross_modal_relationship(
    file_metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build the logical relationship between one optical observation
    and one SAR observation.

    The physical configuration itself establishes that this is a
    cross-modal request:

        optical
        +
        SAR(VV,VH)

    Relationship-level metadata such as co-registration is optional.
    It is preserved when consistently supplied by trusted metadata.
    """

    metadata = file_metadata

    relationship = {
        "type": "cross_modal",
    }

    relationship_type_values = {
        item.get("relationship_type")
        for item in metadata
        if item.get("relationship_type") is not None
    }

    # Explicit relationship metadata may exist, but it must not
    # contradict the structurally determined cross-modal relationship.
    if relationship_type_values and relationship_type_values != {
        "cross_modal"
    }:
        raise InputManifestCompatibilityError(_SAFE_ERROR)

    co_registered = _uniform_boolean(
        metadata,
        "co_registered",
    )

    if co_registered is not None:
        relationship["co_registered"] = co_registered

    same_geographic_area = _uniform_boolean(
        metadata,
        "same_geographic_area",
    )

    if same_geographic_area is not None:
        relationship["same_geographic_area"] = same_geographic_area

    return relationship


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_temporal_order(
    observations: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
) -> None:
    """
    Ensure two optical observations have distinct timestamps.

    The images are allowed to arrive in arbitrary upload order.
    """

    timestamps = [
        _normalize_timestamp(
            item.get("acquisition_time")
        )
        for item in metadata
    ]

    if any(timestamp is None for timestamp in timestamps):
        raise InputManifestCompatibilityError(_SAFE_ERROR)

    if len(set(timestamps)) != len(timestamps):
        raise InputManifestCompatibilityError(
            "Temporal observations must have distinct acquisition times."
        )


def _validate_temporal_observations(
    observations: list[dict[str, Any]],
) -> None:
    timestamps = [
        observation.get("acquisition_time")
        for observation in observations
    ]

    if any(timestamp is None for timestamp in timestamps):
        raise InputManifestCompatibilityError(_SAFE_ERROR)

    if len(set(timestamps)) != len(timestamps):
        raise InputManifestCompatibilityError(
            "Temporal observations must have distinct acquisition times."
        )


def _uniform_boolean(
    metadata: list[dict[str, Any]],
    key: str,
) -> bool | None:
    """
    Return a boolean only when the supplied relationship metadata
    is explicit and consistent.

    Examples:

        [None, None, None] -> None
        [True, True, True] -> True
        [False, False, False] -> False
        [True, None, True] -> None
        [True, False, True] -> None

    Absence or partial presence does not make the underlying image
    combination invalid.
    """

    values = [
        item.get(key)
        for item in metadata
        if item.get(key) is not None
    ]

    if not values:
        return None

    normalized = set(values)

    if normalized == {True}:
        return True

    if normalized == {False}:
        return False

    # Conflicting or malformed relationship metadata.
    raise InputManifestCompatibilityError(_SAFE_ERROR)


def _normal_modality(
    value: Any,
) -> str | None:
    if not isinstance(value, str):
        return None

    value = value.strip().lower()

    if value not in _SUPPORTED_MODALITIES:
        return None

    return value


def _polarization(
    value: Any,
) -> str | None:
    if not isinstance(value, str):
        return None

    value = value.strip().upper()

    if value.startswith("SAR_"):
        value = value[4:]

    if value not in _SUPPORTED_POLARIZATIONS:
        return None

    return value


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


def _normalize_timestamp(
    value: Any,
) -> str | None:
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.isoformat()

    if not isinstance(value, str):
        return None

    normalized = value.strip()

    return normalized or None


def _parse_timestamp(
    value: str | None,
) -> datetime | None:
    if not value:
        return None

    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except (
        TypeError,
        ValueError,
    ):
        return None


# ---------------------------------------------------------------------------
# Manifest assembly
# ---------------------------------------------------------------------------


def _manifest(
    files: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    relationship: dict[str, Any],
) -> dict[str, Any]:
    return {
        "physical_files": files,
        "observations": observations,
        "relationship": relationship,
    }