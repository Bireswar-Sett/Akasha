import pytest

from qwen.controller.schemas import build_input_manifest


def urls(count):
    return [f"https://storage.example/file-{index}.tiff" for index in range(count)]


def manifest(files, observations, relationship):
    return {
        "physical_files": [{"id": f"file_{index}", "role": role} for index, role in enumerate(files)],
        "observations": observations,
        "relationship": relationship,
    }


def test_single_optical():
    result = build_input_manifest(urls(1), raw_manifest=manifest(["optical"], [
        {"id": "obs_1", "modality": "optical", "image": {"id": "file_0"}},
    ], {"type": "single"}))
    assert len(result.physical_files) == 1
    assert len(result.observations) == 1
    assert result.observations[0].modality == "optical"


def test_single_sar_is_one_observation():
    result = build_input_manifest(urls(2), raw_manifest=manifest(["sar_vv", "sar_vh"], [
        {"id": "obs_1", "modality": "sar", "sar": {"vv": {"id": "file_0"}, "vh": {"id": "file_1"}}},
    ], {"type": "single"}))
    assert len(result.physical_files) == 2
    assert len(result.observations) == 1
    assert result.observations[0].sar.vv.image_id == "file_0"
    assert result.observations[0].sar.vh.image_id == "file_1"


def test_two_optical_observations_are_explicit():
    result = build_input_manifest(urls(2), raw_manifest=manifest(["optical", "optical"], [
        {"id": "t1", "modality": "optical", "acquisition_time": "2024-01-01", "image": {"id": "file_0"}},
        {"id": "t2", "modality": "optical", "acquisition_time": "2025-01-01", "image": {"id": "file_1"}},
    ], {"type": "bi_temporal"}))
    assert len(result.observations) == 2


def test_optical_plus_sar():
    result = build_input_manifest(urls(3), raw_manifest=manifest(["optical", "sar_vv", "sar_vh"], [
        {"id": "optical", "modality": "optical", "image": {"id": "file_0"}},
        {"id": "sar", "modality": "sar", "sar": {"vv": {"id": "file_1"}, "vh": {"id": "file_2"}}},
    ], {"type": "cross_modal", "co_registered": True}))
    assert [item.modality for item in result.observations] == ["optical", "sar"]


def test_two_sar_observations():
    result = build_input_manifest(urls(4), raw_manifest=manifest(["sar_vv", "sar_vh", "sar_vv", "sar_vh"], [
        {"id": "t1", "modality": "sar", "acquisition_time": "2024-01-01", "sar": {"vv": {"id": "file_0"}, "vh": {"id": "file_1"}}},
        {"id": "t2", "modality": "sar", "acquisition_time": "2025-01-01", "sar": {"vv": {"id": "file_2"}, "vh": {"id": "file_3"}}},
    ], {"type": "bi_temporal", "spatially_corresponding": True}))
    assert len(result.observations) == 2
    assert all(item.modality == "sar" for item in result.observations)


def test_fifth_file_is_rejected():
    with pytest.raises(ValueError, match="four"):
        build_input_manifest(urls(5), raw_manifest={"observations": []})


def test_ambiguous_multiple_files_are_rejected():
    with pytest.raises(ValueError, match="backend-generated input manifest"):
        build_input_manifest(urls(2))
