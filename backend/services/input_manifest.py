"""Trusted backend construction of Qwen-compatible input manifests."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


class InputManifestCompatibilityError(ValueError):
    """Raised when Storage metadata cannot safely describe the uploads."""


_SAFE_ERROR = "The uploaded files cannot be safely grouped into supported observations. Please upload a supported imagery combination."


def build_input_manifest(file_metadata: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the sole production physical-file -> observation mapping.

    Values come exclusively from Firebase Storage metadata.  In particular,
    this function never examines a path or filename to infer channel roles.
    """
    if not 1 <= len(file_metadata) <= 4:
        raise InputManifestCompatibilityError("between one and four image files are supported")
    if any(not isinstance(item, dict) for item in file_metadata):
        raise InputManifestCompatibilityError(_SAFE_ERROR)

    files = [_physical_file(index, item) for index, item in enumerate(file_metadata)]
    modalities = [_normal_modality(item.get("modality")) for item in file_metadata]
    if any(modality is None for modality in modalities):
        raise InputManifestCompatibilityError(_SAFE_ERROR)

    if len(files) == 1:
        if modalities[0] not in {"optical", "multispectral"}:
            raise InputManifestCompatibilityError(_SAFE_ERROR)
        return _manifest(files, [_image_observation("observation_1", modalities[0], files[0], file_metadata[0])], {"type": "single"})

    sar_indices = [index for index, modality in enumerate(modalities) if modality == "sar"]
    optical_indices = [index for index, modality in enumerate(modalities) if modality in {"optical", "multispectral"}]

    if len(files) == 2 and len(optical_indices) == 2:
        relationship = _temporal_relationship(file_metadata)
        observations = [_image_observation(f"observation_{index + 1}", modalities[index], files[index], file_metadata[index]) for index in optical_indices]
        return _manifest(files, observations, relationship)

    if len(files) == 2 and len(sar_indices) == 2:
        observation = _sar_observation("observation_1", sar_indices, files, file_metadata)
        return _manifest(files, [observation], {"type": "single"})

    if len(files) == 3 and len(optical_indices) == 1 and len(sar_indices) == 2:
        relationship = _cross_modal_relationship(file_metadata)
        optical_index = optical_indices[0]
        observations = [
            _image_observation("observation_optical", modalities[optical_index], files[optical_index], file_metadata[optical_index]),
            _sar_observation("observation_sar", sar_indices, files, file_metadata),
        ]
        return _manifest(files, observations, relationship)

    if len(files) == 4 and len(sar_indices) == 4:
        relationship = _temporal_relationship(file_metadata)
        grouped = _sar_groups(sar_indices, file_metadata)
        if len(grouped) != 2:
            raise InputManifestCompatibilityError(_SAFE_ERROR)
        observations = [_sar_observation(f"observation_{index + 1}", group, files, file_metadata) for index, group in enumerate(grouped)]
        return _manifest(files, observations, relationship)

    raise InputManifestCompatibilityError(_SAFE_ERROR)


def _physical_file(index: int, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"file_{index}",
        **{key: metadata.get(key) for key in ("filename", "format", "size", "role", "modality", "acquisition_time", "bands", "polarization") if metadata.get(key) is not None},
    }


def _image_observation(observation_id: str, modality: str, file: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    observation = {"id": observation_id, "modality": modality, "image": {"id": file["id"]}}
    if metadata.get("acquisition_time") is not None:
        observation["acquisition_time"] = metadata["acquisition_time"]
    return observation


def _sar_observation(observation_id: str, indices: list[int], files: list[dict[str, Any]], metadata: list[dict[str, Any]]) -> dict[str, Any]:
    if len(indices) != 2:
        raise InputManifestCompatibilityError(_SAFE_ERROR)
    roles = {_polarization(metadata[index].get("polarization") or metadata[index].get("role")): index for index in indices}
    if set(roles) != {"VV", "VH"} or len(roles) != 2:
        raise InputManifestCompatibilityError(_SAFE_ERROR)
    timestamps = {metadata[index].get("acquisition_time") for index in indices}
    if len(timestamps) != 1:
        raise InputManifestCompatibilityError(_SAFE_ERROR)
    observation = {
        "id": observation_id,
        "modality": "sar",
        "sar": {"vv": {"id": files[roles["VV"]]["id"]}, "vh": {"id": files[roles["VH"]]["id"]}},
    }
    timestamp = timestamps.pop()
    if timestamp is not None:
        observation["acquisition_time"] = timestamp
    return observation


def _sar_groups(indices: list[int], metadata: list[dict[str, Any]]) -> list[list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        observation_id = metadata[index].get("observation_id")
        if not isinstance(observation_id, str) or not observation_id.strip():
            raise InputManifestCompatibilityError(_SAFE_ERROR)
        groups[observation_id].append(index)
    return list(groups.values())


def _temporal_relationship(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    if _consistent(metadata, "relationship_type") not in {"bi_temporal", "temporal"}:
        raise InputManifestCompatibilityError(_SAFE_ERROR)
    if not _all_true(metadata, "spatially_corresponding"):
        raise InputManifestCompatibilityError(_SAFE_ERROR)
    if any(item.get("acquisition_time") is None for item in metadata):
        raise InputManifestCompatibilityError(_SAFE_ERROR)
    return {"type": "bi_temporal", "spatially_corresponding": True}


def _cross_modal_relationship(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    if _consistent(metadata, "relationship_type") != "cross_modal" or not _all_true(metadata, "co_registered"):
        raise InputManifestCompatibilityError(_SAFE_ERROR)
    relationship = {"type": "cross_modal", "co_registered": True}
    if _all_true(metadata, "same_geographic_area"):
        relationship["same_geographic_area"] = True
    return relationship


def _consistent(metadata: list[dict[str, Any]], key: str) -> Any:
    values = {item.get(key) for item in metadata}
    return values.pop() if len(values) == 1 else None


def _all_true(metadata: list[dict[str, Any]], key: str) -> bool:
    return all(item.get(key) is True for item in metadata)


def _normal_modality(value: Any) -> str | None:
    value = value.lower().strip() if isinstance(value, str) else ""
    return value if value in {"optical", "multispectral", "sar"} else None


def _polarization(value: Any) -> str | None:
    value = value.upper().strip() if isinstance(value, str) else ""
    if value.startswith("SAR_"):
        value = value[4:]
    return value if value in {"VV", "VH"} else None


def _manifest(files: list[dict[str, Any]], observations: list[dict[str, Any]], relationship: dict[str, Any]) -> dict[str, Any]:
    return {"physical_files": files, "observations": observations, "relationship": relationship}
