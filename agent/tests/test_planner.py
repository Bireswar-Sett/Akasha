import pytest
from pydantic import ValidationError

from qwen.controller.planner import TaskPlanner
from qwen.controller.bbox import BoundingBoxError, parse_query
from qwen.controller.controller import QwenController
from qwen.controller.schemas import AnalysisRequest, InputConfiguration


URLS = [f"https://example.test/image-{index}.tif" for index in range(1, 5)]


def test_single_optical_routes_to_geochat():
    plan = TaskPlanner().plan(AnalysisRequest(user_request="Describe this image", signed_image_urls=[URLS[0]], metadata={"images": [{"id": "image_1", "modality": "optical"}]}))
    assert [call.name for call in plan.calls] == ["geochat"]
    assert "REGION OF INTEREST" not in plan.calls[0].arguments["prompt"]
    assert "Describe the scene" in plan.calls[0].arguments["prompt"]


def test_single_image_vqa_is_valid():
    plan = TaskPlanner().plan(AnalysisRequest(user_request="Is there a railway line?", signed_image_urls=[URLS[0]]))
    assert plan.compatibility_issue is None
    assert [call.name for call in plan.calls] == ["geochat"]


def test_single_image_grounding_is_valid():
    plan = TaskPlanner().plan(AnalysisRequest(user_request="Highlight the water body.", signed_image_urls=[URLS[0]]))
    assert plan.compatibility_issue is None
    assert [call.name for call in plan.calls] == ["geochat"]


def test_single_image_temporal_request_is_incompatible():
    plan = TaskPlanner().plan(AnalysisRequest(user_request="What changed between these two dates?", signed_image_urls=[URLS[0]]))
    assert plan.compatibility_issue["status"] == "input_incompatible"
    assert "two corresponding image inputs" in plan.compatibility_issue["reason"]
    assert plan.calls == []


def test_single_sar_routes_through_pseudo_rgb():
    plan = TaskPlanner().plan(AnalysisRequest(user_request="Describe this image", signed_image_urls=[URLS[0]], metadata={"images": [{"id": "image_1", "modality": "sar"}]}))
    assert [call.name for call in plan.calls] == ["pseudo_rgb", "geochat"]


def test_optical_sar_uses_two_geochat_calls():
    metadata = {"input_configuration": "optical_sar", "images": [{"id": "image_1", "modality": "optical"}, {"id": "image_2", "modality": "sar"}]}
    plan = TaskPlanner().plan(AnalysisRequest(user_request="Compare these images", signed_image_urls=URLS[:2], metadata=metadata))
    assert plan.input_configuration == InputConfiguration.OPTICAL_SAR
    assert [call.name for call in plan.calls] == ["geochat", "pseudo_rgb", "geochat"]


def test_temporal_pair_uses_m2cd():
    metadata = {"input_configuration": "bi_temporal", "images": [{"id": "image_1", "modality": "optical", "acquisition_time": "2024"}, {"id": "image_2", "modality": "optical", "acquisition_time": "2025"}]}
    plan = TaskPlanner().plan(AnalysisRequest(user_request="What changed?", signed_image_urls=URLS[:2], metadata=metadata))
    assert [call.name for call in plan.calls] == ["m2cd"]


def test_four_url_sar_edge_case_is_valid():
    metadata = {"sar_channels": ["vv_t1", "vh_t1", "vv_t2", "vh_t2"]}
    plan = TaskPlanner().plan(AnalysisRequest(user_request="What changed?", signed_image_urls=URLS, metadata=metadata))
    assert plan.compatibility_issue is None
    assert [call.name for call in plan.calls] == ["m2cd"]


def test_three_images_reach_compatibility_validation():
    plan = TaskPlanner().plan(AnalysisRequest(user_request="Analyze these images", signed_image_urls=URLS[:3]))
    assert plan.compatibility_issue is not None
    assert "relationship" in plan.compatibility_issue["reason"]


def test_ambiguous_pair_is_rejected_without_guessing():
    plan = TaskPlanner().plan(AnalysisRequest(user_request="Analyze these images", signed_image_urls=URLS[:2]))
    assert plan.compatibility_issue is not None
    assert plan.calls == []


def test_url_contract_rejects_zero_more_than_four_and_non_https():
    with pytest.raises(ValidationError):
        AnalysisRequest(user_request="Describe", signed_image_urls=[])
    with pytest.raises(ValidationError):
        AnalysisRequest(user_request="Describe", signed_image_urls=URLS + ["https://example.test/5.tif"])
    with pytest.raises(ValidationError):
        AnalysisRequest(user_request="Describe", signed_image_urls=["/tmp/image.tif"])


def test_bbox_parsing_preserves_original_question_and_rotation():
    parsed = parse_query("What is visible here?\nb = {20, 35, 70, 80|15}")
    assert parsed.user_query == "What is visible here?"
    assert parsed.bounding_boxes[0].angle == 15
    assert parsed.bounding_boxes[0].compact() == "{20, 35, 70, 80|15}"


@pytest.mark.parametrize("query", [
    "Describe\nb = {20, 35, 70, 80}",
    "Describe\nb = {101, 0, 70, 80|0}",
    "Describe\nb = {70, 35, 20, 80|0}",
])
def test_invalid_bbox_is_rejected(query):
    with pytest.raises((BoundingBoxError, ValidationError)):
        AnalysisRequest(user_request=query, signed_image_urls=[URLS[0]])


def test_bbox_routes_single_image_to_region_geochat():
    request = AnalysisRequest(user_request="What is visible? b = {20, 30, 70, 80|0}", signed_image_urls=[URLS[0]])
    plan = TaskPlanner().plan(request)
    assert [call.name for call in plan.calls] == ["geochat"]
    assert plan.calls[0].operation == "region_grounded_analysis"
    assert plan.calls[0].arguments["bounding_box"]["x_left"] == 20
    assert "{20, 30, 70, 80|0}" in plan.calls[0].arguments["prompt"]
    assert plan.calls[0].arguments["prompt"] != "Describe the scene and its major visible land-cover types and features."


def test_bbox_prompt_is_task_aware_for_visual_question():
    request = AnalysisRequest(user_request="Is there a railway line? b = {20, 30, 70, 80|15}", signed_image_urls=[URLS[0]])
    prompt = TaskPlanner().plan(request).calls[0].arguments["prompt"]
    assert "Answer the visual question" in prompt
    assert "{20, 30, 70, 80|15}" in prompt
    assert "unrelated image areas" in prompt


def test_multiple_bbox_associations_are_not_collapsed():
    query = "Compare the regions. b = {10, 20, 40, 50|0} b = {60, 25, 90, 70|5}"
    request = AnalysisRequest(user_request=query, signed_image_urls=URLS[:2], metadata={"input_configuration": "optical_sar", "images": [{"id": "image_1", "modality": "optical"}, {"id": "image_2", "modality": "optical"}]})
    plan = TaskPlanner().plan(request)
    assert len(plan.bounding_boxes) == 2
    assert [call.arguments["bounding_box"]["x_left"] for call in plan.calls] == [10, 60]
    assert all(call.name == "geochat" for call in plan.calls)


def test_bbox_temporal_plan_keeps_m2cd_and_geochat_region_calls():
    request = AnalysisRequest(user_request="What changed? b = {20, 30, 70, 80|0} b = {22, 32, 72, 82|5}", signed_image_urls=URLS[:2], metadata={"input_configuration": "bi_temporal"})
    plan = TaskPlanner().plan(request)
    assert [call.name for call in plan.calls] == ["m2cd", "geochat", "geochat"]
    assert all(call.name != "teochat" for call in plan.calls)


def test_controller_preserves_bbox_in_trace_and_evidence():
    class FakeExecutor:
        def execute(self, name, arguments):
            return {"ok": True, "result": {"observation": "similar visible structure"}}

    class FakeQwen:
        def chat(self, messages, **kwargs):
            return "The evidence suggests little visible change."

    request = AnalysisRequest(user_request="Describe this region. b = {20, 30, 70, 80|15}", signed_image_urls=[URLS[0]])
    result = QwenController(FakeQwen(), FakeExecutor()).run_request(request)
    assert result["execution"][0]["operation"] == "region_grounded_analysis"
    assert result["execution"][0]["bounding_boxes"][0]["angle"] == 15
    assert result["evidence"][0]["bounding_boxes"][0]["x_left"] == 20
    assert result["answer"].startswith("The evidence")
