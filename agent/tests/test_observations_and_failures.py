import logging

import pytest
from pydantic import ValidationError

from qwen.controller.controller import QwenController
from qwen.controller.executor import ToolExecutionError, ToolExecutor
from qwen.controller.planner import TaskPlanner
from qwen.controller.schemas import (
    AnalysisRequest,
    ImageReference,
    InputConfiguration,
    InputManifest,
    Observation,
    RelationshipMetadata,
    SARFiles,
    ToolPlan,
)


URLS = [f"https://example.test/{index}.tif" for index in range(1, 5)]


def reference(name, url):
    return ImageReference(image_id=name, url=url)


def optical_observation(name, url, acquisition_time=None):
    return Observation(
        id=name,
        modality="optical",
        acquisition_time=acquisition_time,
        image=reference(f"{name}_image", url),
    )


def sar_observation(name, vv_url, vh_url, acquisition_time=None):
    return Observation(
        id=name,
        modality="sar",
        acquisition_time=acquisition_time,
        sar=SARFiles(
            vv=reference(f"{name}_vv", vv_url),
            vh=reference(f"{name}_vh", vh_url),
        ),
    )


def request_for(manifest, query="Describe the imagery"):
    return AnalysisRequest(
        user_request=query,
        signed_image_urls=[item.url for item in manifest.physical_files],
        manifest=manifest,
    )


def test_manifest_counts_optical_and_sar_as_two_observations_three_files():
    manifest = InputManifest(observations=[
        optical_observation("optical_t1", URLS[0]),
        sar_observation("sar_t1", URLS[1], URLS[2]),
    ], relationship=RelationshipMetadata(relationship="cross_modal"))
    request = request_for(manifest)

    assert len(request.manifest.observations) == 2
    assert len(request.manifest.physical_files) == 3
    assert TaskPlanner().plan(request).input_configuration == InputConfiguration.OPTICAL_SAR


def test_manifest_counts_two_sar_observations_as_four_files():
    manifest = InputManifest(observations=[
        sar_observation("sar_t1", URLS[0], URLS[1], "2024-01-01"),
        sar_observation("sar_t2", URLS[2], URLS[3], "2025-01-01"),
    ], relationship=RelationshipMetadata(relationship="temporal", spatially_corresponding=True))
    request = request_for(manifest, "What changed between the dates?")

    plan = TaskPlanner().plan(request)
    assert len(request.manifest.observations) == 2
    assert len(request.manifest.physical_files) == 4
    assert [call.name for call in plan.calls] == ["m2cd"]
    assert plan.calls[0].arguments == {
        "observation_1_id": "sar_t1",
        "observation_2_id": "sar_t2",
    }


def test_manifest_rejects_sar_without_vv_and_vh():
    with pytest.raises(ValidationError):
        Observation(
            id="sar",
            modality="sar",
            image=reference("image", URLS[0]),
        )


def test_four_physical_files_two_logical_sar_observations_with_null_legacy_modality():
    request = AnalysisRequest(
        user_request="difference between these",
        signed_image_urls=URLS,
        metadata={
            "observations": [
                {"id": "obs_t1", "modality": None, "acquisition_time": "2024-01-04", "sar": {"vv": {"id": "image_1"}, "vh": {"id": "image_2"}}},
                {"id": "obs_t2", "modality": None, "acquisition_time": "2024-04-05", "sar": {"vv": {"id": "image_3"}, "vh": {"id": "image_4"}}},
            ],
            "relationship": {"type": "bi_temporal", "spatially_corresponding": True},
        },
    )
    assert len(request.manifest.observations) == 2
    assert [item.modality for item in request.manifest.observations] == ["sar", "sar"]
    assert all(item.sar and item.sar.vv and item.sar.vh for item in request.manifest.observations)
    plan = TaskPlanner().plan(request)
    assert plan.images_used == ["obs_t1", "obs_t2"]
    assert plan.input_configuration == InputConfiguration.DUAL_SAR


def test_temporal_relationship_requires_two_dated_observations():
    with pytest.raises(ValidationError, match="temporal relationships require exactly two"):
        InputManifest(
            observations=[optical_observation("t1", URLS[0], "2024-01-01")],
            relationship=RelationshipMetadata(type="bi_temporal"),
        )


def test_tool_plan_rejects_unsupported_tools():
    with pytest.raises(ValidationError):
        ToolPlan.model_validate({
            "task_type": "visual_question_answering",
            "task_description": "test",
            "input_configuration": "single_image",
            "images_used": ["observation_1"],
            "calls": [{
                "name": "invented_tool",
                "purpose": "test",
                "operation": "test",
                "arguments": {},
            }],
        })


def test_controller_rejects_manifest_image_ids_not_authorized_by_backend():
    with pytest.raises(ValueError, match="unauthorized image ID"):
        QwenController._authorized_manifest({
            "observations": [{
                "id": "observation_1",
                "modality": "optical",
                "image": "invented_image",
            }],
        }, {"real_image": URLS[0]})


class FailingExecutor:
    def __init__(self, failed_tool):
        self.failed_tool = failed_tool
        self.calls = []

    def execute(self, name, arguments):
        self.calls.append(name)
        if name == self.failed_tool:
            return {"ok": False, "tool": name, "status": "unavailable", "error": f"{name} specialist is currently unavailable."}
        return {"ok": True, "tool": name, "result": {"observation": "supported evidence"}}


class FakeQwen:
    def chat(self, messages, **kwargs):
        return "The requested specialist result was unavailable, so no unsupported conclusion was made."


def test_failed_sar_preprocessing_skips_dependent_geochat():
    manifest = InputManifest(observations=[sar_observation("sar", URLS[0], URLS[1])])
    executor = FailingExecutor("pseudo_rgb")
    result = QwenController(FakeQwen(), executor).run_request(request_for(manifest))

    assert executor.calls == ["pseudo_rgb"]
    assert result["execution"][0]["status"] == "unavailable"
    assert result["execution"][1]["status"] == "skipped"
    assert result["evidence"][1]["error_summary"]


def test_geochat_unavailable_is_transparent():
    manifest = InputManifest(observations=[optical_observation("optical", URLS[0])])
    result = QwenController(FakeQwen(), FailingExecutor("geochat")).run_request(request_for(manifest))

    assert result["status"] == "completed_with_warnings"
    assert result["evidence"][0]["status"] == "unavailable"
    assert "unavailable" in result["evidence"][0]["error_summary"]


def test_teochat_unavailable_is_transparent():
    manifest = InputManifest(observations=[
        optical_observation("t1", URLS[0], "2024-01-01"),
        optical_observation("t2", URLS[1], "2025-01-01"),
    ], relationship=RelationshipMetadata(relationship="temporal", spatially_corresponding=True))
    result = QwenController(FakeQwen(), FailingExecutor("teochat")).run_request(request_for(manifest, "What changed?"))

    assert result["execution"][0]["tool"] == "teochat"
    assert result["execution"][0]["status"] == "unavailable"


def test_m2cd_unavailable_does_not_claim_change_detection():
    manifest = InputManifest(observations=[
        sar_observation("t1", URLS[0], URLS[1], "2024-01-01"),
        sar_observation("t2", URLS[2], URLS[3], "2025-01-01"),
    ], relationship=RelationshipMetadata(relationship="temporal", spatially_corresponding=True))
    result = QwenController(FakeQwen(), FailingExecutor("m2cd")).run_request(request_for(manifest, "What changed after the flood?"))

    assert result["execution"][0]["tool"] == "m2cd"
    assert result["execution"][0]["status"] == "unavailable"
    assert result["evidence"][0]["success"] is False
    assert "detected" not in result["evidence"][0]["error_summary"].lower()


def test_signed_urls_are_not_written_to_logs(caplog):
    signed_url = "https://storage.example/private/signed-secret-token.tif"
    executor = ToolExecutor(hf_token="test-token")

    def fail_remote(*args, **kwargs):
        raise ToolExecutionError("GeoChat specialist is currently unavailable.", "unavailable")

    executor._remote_predict = fail_remote
    with caplog.at_level(logging.WARNING):
        executor.execute("geochat", {"image_ref": signed_url, "prompt": "Describe"})

    assert signed_url not in caplog.text
