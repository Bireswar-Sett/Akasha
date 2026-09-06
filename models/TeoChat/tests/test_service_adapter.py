from __future__ import annotations

import json

from bbox_parser import BoundingBoxError, extract_bboxes, normalize_bboxes
from response_adapter import normalize_response


def test_legacy_bbox_is_normalized_with_image_association():
    result = normalize_bboxes("Image 2: [20, 30, 70, 80]", image_count=2)
    assert result == [{"image_index": 1, "box": {"x_left": 20.0, "y_top": 30.0, "x_right": 70.0, "y_bottom": 80.0, "angle": 0.0}}]


def test_multiple_boxes_and_rotated_canonical_box_are_preserved():
    result = normalize_bboxes("Image 1: [1, 2, 3, 4] Image 2: {10,20,50,60|15}", image_count=2)
    assert len(result) == 2
    assert result[0]["image_index"] == 0
    assert result[1]["image_index"] == 1
    assert result[1]["box"]["angle"] == 15


def test_legacy_evaluation_extractor_remains_compatible():
    assert extract_bboxes("objects [1, 2, 3, 4]") == [[1, 2, 3, 4]]


def test_no_bbox_is_valid_json_response():
    response = normalize_response("No supported region was identified.", image_count=2)
    json.dumps(response)
    assert response["bounding_boxes"] == []
    assert response["evidence"]["confidence"] is None


def test_invalid_bbox_becomes_uncertainty_not_fake_evidence():
    response = normalize_response("Image 1: [90, 20, 10, 80]", image_count=1)
    assert response["bounding_boxes"] == []
    assert response["evidence"]["uncertainty"]
