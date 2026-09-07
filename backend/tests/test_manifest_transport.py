from fastapi import HTTPException
import pytest

from services.qwen_service import QwenRequestBuilder


def test_two_file_sar_manifest_is_transmitted_without_urls_in_manifest():
    request = QwenRequestBuilder.build(
        "Describe this SAR image",
        ["https://storage.example/vv", "https://storage.example/vh"],
        manifest={
            "physical_files": [{"id": "file_0", "role": "sar_vv"}, {"id": "file_1", "role": "sar_vh"}],
            "observations": [{"id": "obs_1", "modality": "sar", "sar": {"vv": {"file_id": "file_0"}, "vh": {"file_id": "file_1"}}}],
            "relationship": {"type": "single"},
        },
    )
    assert len(request.images) == 2
    assert len(request.metadata["physical_files"]) == 2
    assert len(request.metadata["observations"]) == 1
    assert request.metadata["observations"][0]["sar"]["vv"]["id"] == "file_0"
    assert "url" not in request.metadata["physical_files"][0]


def test_multiple_urls_are_grouped_from_trusted_metadata_without_client_manifest():
    request = QwenRequestBuilder.build(
        "Describe this SAR image",
        ["https://storage.example/vv", "https://storage.example/vh"],
        image_metadata=[
            {"modality": "sar", "polarization": "VV"},
            {"modality": "sar", "polarization": "VH"},
        ],
    )
    assert len(request.metadata["observations"]) == 1
    assert request.metadata["observations"][0]["modality"] == "sar"
