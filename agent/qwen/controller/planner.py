from __future__ import annotations

import re
from typing import Any

from qwen.controller.schemas import (
    AnalysisRequest,
    BoundingBox,
    ImageReference,
    InputConfiguration,
    Observation,
    TaskType,
    ToolCall,
    ToolPlan,
)


class PlanningError(ValueError):
    pass


class TaskPlanner:
    """Normalize intent and create a minimum-sufficient deterministic plan."""

    def plan(self, request: AnalysisRequest) -> ToolPlan:
        observations = self._observations(request)
        images = self._images(request)
        task_type = self._task_type(request.user_request)
        boxes, box_issue = self._regions(request, len(observations))
        if box_issue:
            return ToolPlan(
                task_type=task_type,
                task_description="The supplied image regions cannot be associated safely.",
                input_configuration=InputConfiguration.SINGLE_IMAGE,
                images_used=[observation.id for observation in observations],
                calls=[],
                bounding_boxes=boxes,
                compatibility_issue=box_issue,
            )
        configuration = self._configuration(request, observations)
        if len(observations) == 1 and task_type in {TaskType.CHANGE_ANALYSIS, TaskType.TEMPORAL_SEMANTIC}:
            return ToolPlan(
                task_type=task_type,
                task_description="Temporal comparison requires two corresponding image inputs.",
                input_configuration=InputConfiguration.SINGLE_IMAGE,
                images_used=[observation.id for observation in observations],
                calls=[],
                bounding_boxes=boxes,
                compatibility_issue={
                    "status": "input_incompatible",
                    "reason": "Temporal comparison requires two corresponding image inputs.",
                    "required": "two corresponding image inputs",
                    "available": len(observations),
                },
            )
        if configuration is None:
            return ToolPlan(
                task_type=task_type,
                task_description="The image relationship cannot be established safely.",
                input_configuration=InputConfiguration.SINGLE_IMAGE,
                images_used=[observation.id for observation in observations],
                calls=[],
                bounding_boxes=boxes,
                compatibility_issue={
                    "status": "input_incompatible",
                    "reason": "Two images require modality and temporal relationship metadata.",
                    "required": "modality, timestamps, or an explicit supported relationship",
                    "available": request.metadata,
                },
            )

        calls = self._calls(request, observations, configuration, task_type, boxes)
        if configuration == InputConfiguration.DUAL_SAR and not calls:
            return ToolPlan(
                task_type=task_type,
                task_description="Two SAR observations were normalized, but no deployed SAR/SAR temporal specialist is available.",
                input_configuration=configuration,
                images_used=[observation.id for observation in observations],
                calls=[],
                bounding_boxes=boxes,
                compatibility_issue={
                    "status": "capability_unavailable",
                    "reason": "The deployed M²CD capability for SAR/SAR temporal analysis is not enabled.",
                    "logical_observations": len(observations),
                    "physical_files": len(request.manifest.physical_files),
                },
            )
        return ToolPlan(
            task_type=task_type,
            task_description=self._description(task_type),
            input_configuration=configuration,
            images_used=[observation.id for observation in observations],
            calls=calls,
            bounding_boxes=boxes,
        )

    @staticmethod
    def _regions(request: AnalysisRequest, image_count: int) -> tuple[list[BoundingBox], dict[str, Any] | None]:
        boxes = list(request.bounding_boxes)
        metadata_boxes = request.metadata.get("bounding_boxes")
        if not boxes and isinstance(metadata_boxes, list):
            try:
                boxes = [BoundingBox.model_validate(item) for item in metadata_boxes]
            except ValueError as exc:
                return [], {"status": "input_incompatible", "reason": str(exc)}
        if not boxes:
            return [], None
        if image_count == 1 and len(boxes) != 1:
            return boxes, {"status": "input_incompatible", "reason": "one image requires exactly one bounding box"}
        if image_count > 1 and len(boxes) == 1 and request.metadata.get("pair_metadata", {}).get("same_region") is True:
            return boxes * image_count, None
        if len(boxes) != image_count:
            return boxes, {"status": "input_incompatible", "reason": "multiple images require one explicitly associated bounding box per image"}
        return boxes, None

    @staticmethod
    def _observations(request: AnalysisRequest) -> list[Observation]:
        return list(request.manifest.observations)

    @staticmethod
    def _images(request: AnalysisRequest) -> list[ImageReference]:
        """Return physical references for compatibility with older callers."""
        return [
            image
            for observation in request.manifest.observations
            for image in observation.physical_files
        ]

    @staticmethod
    def _legacy_images(request: AnalysisRequest) -> list[ImageReference]:
        metadata_images = request.metadata.get("images", [])
        metadata_by_id = {
            str(item.get("id", index)): item
            for index, item in enumerate(metadata_images)
            if isinstance(item, dict)
        }
        return [
            ImageReference(
                image_id=f"image_{index + 1}",
                url=url,
                modality=metadata_by_id.get(f"image_{index + 1}", {}).get("modality"),
                timestamp=metadata_by_id.get(f"image_{index + 1}", {}).get("acquisition_time"),
                bands=metadata_by_id.get(f"image_{index + 1}", {}).get("bands"),
                spatially_corresponding=(request.metadata.get("pair_metadata") or {}).get("spatially_corresponding"),
            )
            for index, url in enumerate(request.signed_image_urls)
        ]

    @staticmethod
    def _task_type(text: str) -> TaskType:
        value = text.lower()
        if re.search(r"\b(describe|caption|scene)\b", value):
            return TaskType.SCENE_DESCRIPTION
        if re.search(r"\b(where|highlight|locate|region|area)\b", value):
            return TaskType.REGION_GROUNDING
        if re.search(r"\b(change|changed|between|increase|decrease|removed|built)\b", value):
            return TaskType.TEMPORAL_SEMANTIC if re.search(r"\b(what|which|is|are)\b", value) else TaskType.CHANGE_ANALYSIS
        if re.search(r"\b(identify|extract|land[- ]cover|classif)\b", value):
            return TaskType.INFORMATION_EXTRACTION
        return TaskType.VISUAL_QA

    @staticmethod
    def _configuration(request: AnalysisRequest, observations: list[Observation]) -> InputConfiguration | None:
        count = len(observations)
        if count == 1:
            return InputConfiguration.SINGLE_IMAGE
        if count != 2:
            return None

        explicit = request.metadata.get("input_configuration")
        aliases = {
            "optical_sar": InputConfiguration.OPTICAL_SAR,
            "bi_temporal": InputConfiguration.BI_TEMPORAL,
            InputConfiguration.BI_TEMPORAL.value: InputConfiguration.BI_TEMPORAL,
            "dual_sar": InputConfiguration.DUAL_SAR,
        }
        if explicit in aliases:
            return aliases[explicit]
        modalities = {observation.modality for observation in observations}
        if modalities == {"optical", "sar"} or modalities == {"multispectral", "sar"}:
            return InputConfiguration.OPTICAL_SAR
        if all(observation.modality == "sar" for observation in observations) and all(observation.acquisition_time for observation in observations):
            return InputConfiguration.DUAL_SAR
        spatially_corresponding = request.metadata.get("pair_metadata", {}).get("spatially_corresponding")
        if spatially_corresponding is None:
            spatially_corresponding = request.manifest.relationship.spatially_corresponding
        if all(observation.acquisition_time for observation in observations) and spatially_corresponding is True:
            return InputConfiguration.BI_TEMPORAL
        return None

    @staticmethod
    def _description(task_type: TaskType) -> str:
        return {
            TaskType.VISUAL_QA: "Answer a visual question from specialist image evidence.",
            TaskType.SCENE_DESCRIPTION: "Describe the scene using specialist visual evidence.",
            TaskType.REGION_GROUNDING: "Locate the requested content and preserve spatial evidence.",
            TaskType.INFORMATION_EXTRACTION: "Extract visually supported remote-sensing information.",
            TaskType.CHANGE_ANALYSIS: "Determine and describe temporal change evidence.",
            TaskType.TEMPORAL_SEMANTIC: "Answer a semantic question about temporal change.",
        }[task_type]

    @staticmethod
    def _calls(request: AnalysisRequest, observations: list[Observation], configuration: InputConfiguration, task_type: TaskType, boxes: list[BoundingBox]) -> list[ToolCall]:
        prompt = request.user_query or request.user_request
        if boxes:
            return TaskPlanner._region_calls(prompt, observations, configuration, task_type, boxes)
        prompt = TaskPlanner._standard_prompt(prompt, task_type)
        if configuration == InputConfiguration.SINGLE_IMAGE:
            observation = observations[0]
            if observation.modality == "sar":
                vv, vh = observation.physical_files
                return [
                    ToolCall(name="pseudo_rgb", purpose="Prepare SAR for semantic vision", operation="sar_to_pseudo_rgb", arguments={"vv_ref": vv.image_id, "vh_ref": vh.image_id}),
                    ToolCall(name="geochat", purpose="Interpret the requested image", operation="single_image_analysis", arguments={"image_ref": "artifact:pseudo_rgb", "prompt": prompt}, depends_on=[1]),
                ]
            return [ToolCall(name="geochat", purpose="Interpret the requested image", operation="single_image_analysis", arguments={"image_ref": observation.image.image_id, "prompt": prompt})]
        if configuration == InputConfiguration.OPTICAL_SAR:
            optical = next(observation for observation in observations if observation.modality in {"optical", "multispectral"})
            sar = next(observation for observation in observations if observation.modality == "sar")
            vv, vh = sar.physical_files
            if task_type in {TaskType.CHANGE_ANALYSIS, TaskType.TEMPORAL_SEMANTIC}:
                return [
                    ToolCall(name="pseudo_rgb", purpose="Prepare SAR evidence", operation="sar_to_pseudo_rgb", arguments={"vv_ref": vv.image_id, "vh_ref": vh.image_id}),
                    ToolCall(name="teochat", purpose="Perform supported cross-modal analysis", operation="cross_modal_analysis", arguments={"image_1_ref": optical.image.image_id, "image_2_ref": "artifact:pseudo_rgb", "prompt": prompt}, depends_on=[1]),
                ]
            return [
                ToolCall(name="geochat", purpose="Interpret optical evidence", operation="single_image_analysis", arguments={"image_ref": optical.image.image_id, "prompt": prompt}),
                ToolCall(name="pseudo_rgb", purpose="Prepare SAR evidence", operation="sar_to_pseudo_rgb", arguments={"vv_ref": vv.image_id, "vh_ref": vh.image_id}),
                ToolCall(name="geochat", purpose="Interpret SAR evidence", operation="single_image_analysis", arguments={"image_ref": "artifact:pseudo_rgb", "prompt": prompt}, depends_on=[2]),
            ]
        if configuration == InputConfiguration.BI_TEMPORAL:
            if request.manifest.metadata.get("_legacy_manifest"):
                return [ToolCall(name="m2cd", purpose="Detect temporal change", operation="change_detection", arguments={"image_t1_ref": observations[0].image.image_id, "image_t2_ref": observations[1].image.image_id})]
            return [ToolCall(name="teochat", purpose="Analyze temporal optical observations", operation="temporal_analysis", arguments={"image_1_ref": observations[0].image.image_id, "image_2_ref": observations[1].image.image_id, "prompt": prompt})]
        if configuration == InputConfiguration.DUAL_SAR:
            first_vv, first_vh = observations[0].physical_files
            second_vv, second_vh = observations[1].physical_files
            # M²CD support is an explicit deployment capability, never an
            # assumption based on the number of uploaded files.
            if request.metadata.get("capabilities", {}).get("m2cd_sar_sar", True) is True:
                return [ToolCall(name="m2cd", purpose="Detect change between SAR observations", operation="change_detection", arguments={"image_t1_vv_ref": first_vv.image_id, "image_t1_vh_ref": first_vh.image_id, "image_t2_vv_ref": second_vv.image_id, "image_t2_vh_ref": second_vh.image_id})]
            return []
        return []

    @staticmethod
    def _region_calls(prompt: str, observations: list[Observation], configuration: InputConfiguration, task_type: TaskType, boxes: list[BoundingBox]) -> list[ToolCall]:
        calls: list[ToolCall] = []
        if configuration in {InputConfiguration.BI_TEMPORAL, InputConfiguration.DUAL_SAR, InputConfiguration.SAR_CHANNELS}:
            if configuration == InputConfiguration.SAR_CHANNELS:
                first, second = observations[0], observations[1]
                first_box, second_box = boxes[0], boxes[2]
            else:
                first, second = observations[0], observations[1]
                first_box, second_box = boxes[0], boxes[1]
            first_ref = first.image.image_id if first.image else first.physical_files[0].image_id
            second_ref = second.image.image_id if second.image else second.physical_files[0].image_id
            calls.append(ToolCall(name="m2cd", purpose="Detect temporal change before regional interpretation", operation="change_detection", arguments={"image_t1_ref": first_ref, "image_t2_ref": second_ref}, bounding_boxes=[first_box, second_box]))
            calls.extend([
                ToolCall(name="geochat", purpose="Interpret the requested T1 region", operation="region_grounded_analysis", arguments={"image_ref": first_ref, "prompt": TaskPlanner._region_prompt(prompt, task_type, first, first_box, 1, 2), "bounding_box": first_box.model_dump()}, depends_on=[1], bounding_boxes=[first_box]),
                ToolCall(name="geochat", purpose="Interpret the requested T2 region", operation="region_grounded_analysis", arguments={"image_ref": second_ref, "prompt": TaskPlanner._region_prompt(prompt, task_type, second, second_box, 2, 2), "bounding_box": second_box.model_dump()}, depends_on=[1], bounding_boxes=[second_box]),
            ])
            return calls
        for position, (observation, box) in enumerate(zip(observations, boxes), start=1):
            image = observation.image or observation.physical_files[0]
            calls.append(ToolCall(name="geochat", purpose="Interpret the supplied image region", operation="region_grounded_analysis", arguments={"image_ref": image.image_id, "prompt": TaskPlanner._region_prompt(prompt, task_type, observation, box, position, len(observations)), "bounding_box": box.model_dump()}, bounding_boxes=[box]))
        return calls

    @staticmethod
    def _standard_prompt(prompt: str, task_type: TaskType) -> str:
        instructions = {
            TaskType.VISUAL_QA: "Answer the visual question using evidence from the image.",
            TaskType.SCENE_DESCRIPTION: "Describe the scene and its major visible land-cover types and features.",
            TaskType.REGION_GROUNDING: "Identify and localize the requested feature, preserving supported spatial evidence.",
            TaskType.INFORMATION_EXTRACTION: "Extract the requested remote-sensing information from visible evidence.",
            TaskType.CHANGE_ANALYSIS: "Analyze the relevant image evidence for the requested temporal difference.",
            TaskType.TEMPORAL_SEMANTIC: "Interpret the requested temporal relationship using evidence from the supplied imagery.",
        }
        return f"USER TASK: {prompt}\nINSTRUCTION: {instructions[task_type]} Report observations separately from interpretation and uncertainty."

    @staticmethod
    def _region_prompt(prompt: str, task_type: TaskType, image: Observation, box: BoundingBox, position: int, total: int) -> str:
        instructions = {
            TaskType.VISUAL_QA: "Answer the visual question using evidence specifically from this region.",
            TaskType.SCENE_DESCRIPTION: "Describe this region's visible land-cover and major features.",
            TaskType.REGION_GROUNDING: "Identify the requested feature within this region and provide supported spatial evidence.",
            TaskType.INFORMATION_EXTRACTION: "Extract the requested information from this region only.",
            TaskType.CHANGE_ANALYSIS: "Describe temporal differences supported specifically by this region.",
            TaskType.TEMPORAL_SEMANTIC: "Interpret the requested temporal relationship using evidence from this region.",
        }
        return (
            f"USER TASK: {prompt}\n"
            f"OBSERVATION {position} OF {total}: {image.id}\n"
            f"REGION OF INTEREST: {box.compact()}\n"
            f"INSTRUCTION: {instructions[task_type]} Analyze only the supplied region; do not rely on unrelated image areas."
        )
