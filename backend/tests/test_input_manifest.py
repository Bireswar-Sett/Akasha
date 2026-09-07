import pytest

from services.input_manifest import InputManifestCompatibilityError, build_input_manifest


def test_single_sar_is_built_without_client_manifest():
    manifest = build_input_manifest([
        {"modality": "sar", "polarization": "VV", "acquisition_time": "2024-01-01"},
        {"modality": "sar", "polarization": "VH", "acquisition_time": "2024-01-01"},
    ])
    assert len(manifest["physical_files"]) == 2
    assert len(manifest["observations"]) == 1
    assert manifest["observations"][0]["modality"] == "sar"
    assert manifest["observations"][0]["sar"] == {"vv": {"id": "file_0"}, "vh": {"id": "file_1"}}
    assert manifest["relationship"] == {"type": "single"}


def test_optical_and_sar_are_built_from_trusted_relationship_metadata():
    manifest = build_input_manifest([
        {"modality": "optical", "relationship_type": "cross_modal", "co_registered": True},
        {"modality": "sar", "polarization": "VV", "relationship_type": "cross_modal", "co_registered": True},
        {"modality": "sar", "polarization": "VH", "relationship_type": "cross_modal", "co_registered": True},
    ])
    assert [item["modality"] for item in manifest["observations"]] == ["optical", "sar"]
    assert manifest["observations"][1]["sar"]["vh"]["id"] == "file_2"


def test_two_sar_observations_require_explicit_capability_later_not_grouping_guesswork():
    manifest = build_input_manifest([
        {"modality": "sar", "polarization": "VV", "observation_id": "t1", "acquisition_time": "2024-01-01", "relationship_type": "bi_temporal", "spatially_corresponding": True},
        {"modality": "sar", "polarization": "VH", "observation_id": "t1", "acquisition_time": "2024-01-01", "relationship_type": "bi_temporal", "spatially_corresponding": True},
        {"modality": "sar", "polarization": "VV", "observation_id": "t2", "acquisition_time": "2025-01-01", "relationship_type": "bi_temporal", "spatially_corresponding": True},
        {"modality": "sar", "polarization": "VH", "observation_id": "t2", "acquisition_time": "2025-01-01", "relationship_type": "bi_temporal", "spatially_corresponding": True},
    ])
    assert len(manifest["observations"]) == 2
    assert manifest["relationship"]["type"] == "bi_temporal"


@pytest.mark.parametrize("metadata", [
    [{"modality": "sar", "polarization": "VV"}, {"modality": "sar", "polarization": "VV"}],
    [{"modality": "optical"}, {"modality": "optical"}],
    [{"modality": "optical"}] * 5,
])
def test_ambiguous_or_over_limit_inputs_are_rejected(metadata):
    with pytest.raises(InputManifestCompatibilityError):
        build_input_manifest(metadata)
