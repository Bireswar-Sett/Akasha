from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from services.qwen_service import QwenRequestBuilder, QwenService


def test_qwen_request_builder_supports_four_signed_urls():
    request = QwenRequestBuilder.build(
        "What changed?",
        [f"https://storage.example/{index}" for index in range(4)],
        image_metadata=[
            {"modality": "sar", "polarization": "VV", "observation_id": "t1", "acquisition_time": "2024-01-01", "relationship_type": "bi_temporal", "spatially_corresponding": True},
            {"modality": "sar", "polarization": "VH", "observation_id": "t1", "acquisition_time": "2024-01-01", "relationship_type": "bi_temporal", "spatially_corresponding": True},
            {"modality": "sar", "polarization": "VV", "observation_id": "t2", "acquisition_time": "2025-01-01", "relationship_type": "bi_temporal", "spatially_corresponding": True},
            {"modality": "sar", "polarization": "VH", "observation_id": "t2", "acquisition_time": "2025-01-01", "relationship_type": "bi_temporal", "spatially_corresponding": True},
        ],
    )
    assert len(request.images) == 4
    assert request.images[0].metadata["polarization"] == "VV"
    assert request.metadata["relationship"]["type"] == "bi_temporal"
    assert len(request.metadata["observations"]) == 2


def test_qwen_request_builder_rejects_fifth_url():
    with pytest.raises(HTTPException) as error:
        QwenRequestBuilder.build("Analyze", [f"https://storage.example/{index}" for index in range(5)])
    assert error.value.status_code == 400


def test_qwen_request_builder_rejects_empty_url():
    with pytest.raises(HTTPException) as error:
        QwenRequestBuilder.build("Analyze", [""])
    assert error.value.status_code == 400


def test_qwen_service_calls_current_signed_url_interface():
    service = QwenService(space="test/space", token="test-token")
    service._client = MagicMock()
    service._client.predict.return_value = '{"status":"completed","answer":"grounded answer"}'

    result = service.analyze(
        "Describe the image",
        "https://storage.example/one",
        image_metadata=[{"modality": "optical"}],
    )

    assert result == "grounded answer"
    service._client.predict.assert_called_once_with(
        user_request="Describe the image",
        url_1="https://storage.example/one",
        url_2="",
        url_3="",
        url_4="",
        manifest_json='{"physical_files": [{"id": "file_0", "modality": "optical"}], "observations": [{"id": "observation_1", "modality": "optical", "image": {"id": "file_0"}}], "relationship": {"type": "single"}, "metadata": {}, "capabilities": {}}',
        physical_files=[],
        api_name="/analyze",
    )
