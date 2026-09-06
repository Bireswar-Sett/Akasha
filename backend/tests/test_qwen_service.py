from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from services.qwen_service import QwenRequestBuilder, QwenService


def test_qwen_request_builder_supports_four_signed_urls():
    request = QwenRequestBuilder.build(
        "What changed?",
        [f"https://storage.example/{index}" for index in range(4)],
        image_metadata=[{"polarization": role} for role in ("VV", "VH", "VV", "VH")],
        relationship={"relationship": "bi_temporal"},
    )
    assert len(request.images) == 4
    assert request.images[0].metadata["polarization"] == "VV"
    assert request.metadata["pair_metadata"]["relationship"] == "bi_temporal"


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
        image_urls=["https://storage.example/one", "https://storage.example/two"],
    )

    assert result == "grounded answer"
    service._client.predict.assert_called_once_with(
        user_request="Describe the image",
        url_1="https://storage.example/one",
        url_2="https://storage.example/two",
        url_3="",
        url_4="",
        api_name="/analyze",
    )
