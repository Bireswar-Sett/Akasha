from __future__ import annotations

import re
from typing import Any

from qwen.controller.schemas import (
    AnalysisRequest,
    BoundingBox,
    ImageReference,
    InputConfiguration,
    TaskType,
    ToolCall,
    ToolPlan,
)


class PlanningError(ValueError):
    pass


class TaskPlanner:
    """Normalize intent and create a minimum-sufficient deterministic plan."""

    def plan(self, request: AnalysisRequest) -> ToolPlan:
        images = self._images(request)
        task_type = self._task_type(request.user_request)
        boxes, box_issue = self._regions(request, len(images))
        if box_issue:
            return ToolPlan(
                task_type=task_type,
                task_description="The supplied image regions cannot be associated safely.",
                input_configuration=InputConfiguration.SINGLE_IMAGE,
                images_used=[image.image_id for image in images],
                calls=[],
                bounding_boxes=boxes,
                compatibility_issue=box_issue,
            )
        configuration = self._configuration(request, images)
        if len(images) == 1 and task_type in {TaskType.CHANGE_ANALYSIS, TaskType.TEMPORAL_SEMANTIC}:
            return ToolPlan(
                task_type=task_type,
                task_description="Temporal comparison requires two corresponding image inputs.",
                input_configuration=InputConfiguration.SINGLE_IMAGE,
                images_used=[image.image_id for image in images],
                calls=[],
                bounding_boxes=boxes,
                compatibility_issue={
                    "status": "input_incompatible",
                    "reason": "Temporal comparison requires two corresponding image inputs.",
                    "required": "two corresponding image inputs",
                    "available": len(images),
                },
            )
        if configuration is None:
            return ToolPlan(
                task_type=task_type,
                task_description="The image relationship cannot be established safely.",
                input_configuration=InputConfiguration.SINGLE_IMAGE,
                images_used=[image.image_id for image in images],
                calls=[],
                bounding_boxes=boxes,
                compatibility_issue={
                    "status": "input_incompatible",
                    "reason": "Two images require modality and temporal relationship metadata.",
                    "required": "modality, timestamps, or an explicit supported relationship",
                    "available": request.metadata,
                },
            )

        calls = self._calls(request, images, configuration, task_type, boxes)
        return ToolPlan(
            task_type=task_type,
            task_description=self._description(task_type),
            input_configuration=configuration,
            images_used=[image.image_id for image in images],
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
    def _images(request: AnalysisRequest) -> list[ImageReference]:
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
    def _configuration(request: AnalysisRequest, images: list[ImageReference]) -> InputConfiguration | None:
        count = len(images)
        if count == 1:
            return InputConfiguration.SINGLE_IMAGE
        if count == 4:
            roles = request.metadata.get("sar_channels")
            if roles == ["vv_t1", "vh_t1", "vv_t2", "vh_t2"]:
                return InputConfiguration.SAR_CHANNELS
            return None
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
        modalities = {image.modality for image in images}
        if modalities == {"optical", "sar"} or modalities == {"multispectral", "sar"}:
            return InputConfiguration.OPTICAL_SAR
        if all(image.modality == "sar" for image in images) and all(image.timestamp for image in images):
            return InputConfiguration.DUAL_SAR
        if all(image.timestamp for image in images) and request.metadata.get("pair_metadata", {}).get("spatially_corresponding") is True:
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
    def _calls(request: AnalysisRequest, images: list[ImageReference], configuration: InputConfiguration, task_type: TaskType, boxes: list[BoundingBox]) -> list[ToolCall]:
        prompt = request.user_query or request.user_request
        if boxes:
            return TaskPlanner._region_calls(prompt, images, configuration, task_type, boxes)
        prompt = TaskPlanner._standard_prompt(prompt, task_type)
        if configuration == InputConfiguration.SINGLE_IMAGE:
            image = images[0]
            if image.modality == "sar":
                return [
                    ToolCall(name="pseudo_rgb", purpose="Prepare SAR for semantic vision", operation="sar_to_pseudo_rgb", arguments={"image_ref": image.image_id}),
                    ToolCall(name="geochat", purpose="Interpret the requested image", operation="single_image_analysis", arguments={"image_ref": "artifact:pseudo_rgb", "prompt": prompt}, depends_on=[1]),
                ]
            return [ToolCall(name="geochat", purpose="Interpret the requested image", operation="single_image_analysis", arguments={"image_ref": image.image_id, "prompt": prompt})]
        if configuration == InputConfiguration.OPTICAL_SAR:
            optical = next(image for image in images if image.modality in {"optical", "multispectral"})
            sar = next(image for image in images if image.modality == "sar")
            return [
                ToolCall(name="geochat", purpose="Interpret optical evidence", operation="single_image_analysis", arguments={"image_ref": optical.image_id, "prompt": prompt}),
                ToolCall(name="pseudo_rgb", purpose="Prepare SAR evidence", operation="sar_to_pseudo_rgb", arguments={"image_ref": sar.image_id}),
                ToolCall(name="geochat", purpose="Interpret SAR evidence", operation="single_image_analysis", arguments={"image_ref": "artifact:pseudo_rgb", "prompt": prompt}, depends_on=[2]),
            ]
        if configuration in {InputConfiguration.BI_TEMPORAL, InputConfiguration.DUAL_SAR}:
            return [ToolCall(name="m2cd", purpose="Detect temporal change", operation="change_detection", arguments={"image_t1_ref": images[0].image_id, "image_t2_ref": images[1].image_id})]
        if configuration == InputConfiguration.SAR_CHANNELS:
            return [ToolCall(name="m2cd", purpose="Detect temporal change across SAR channels", operation="change_detection", arguments={"image_t1_ref": images[0].image_id, "image_t2_ref": images[2].image_id})]
        return []

    @staticmethod
    def _region_calls(prompt: str, images: list[ImageReference], configuration: InputConfiguration, task_type: TaskType, boxes: list[BoundingBox]) -> list[ToolCall]:
        calls: list[ToolCall] = []
        if configuration in {InputConfiguration.BI_TEMPORAL, InputConfiguration.DUAL_SAR, InputConfiguration.SAR_CHANNELS}:
            if configuration == InputConfiguration.SAR_CHANNELS:
                first, second = images[0], images[2]
                first_box, second_box = boxes[0], boxes[2]
            else:
                first, second = images[0], images[1]
                first_box, second_box = boxes[0], boxes[1]
            calls.append(ToolCall(name="m2cd", purpose="Detect temporal change before regional interpretation", operation="change_detection", arguments={"image_t1_ref": first.image_id, "image_t2_ref": second.image_id}, bounding_boxes=[first_box, second_box]))
            calls.extend([
                ToolCall(name="geochat", purpose="Interpret the requested T1 region", operation="region_grounded_analysis", arguments={"image_ref": first.image_id, "prompt": TaskPlanner._region_prompt(prompt, task_type, first, first_box, 1, 2), "bounding_box": first_box.model_dump()}, depends_on=[1], bounding_boxes=[first_box]),
                ToolCall(name="geochat", purpose="Interpret the requested T2 region", operation="region_grounded_analysis", arguments={"image_ref": second.image_id, "prompt": TaskPlanner._region_prompt(prompt, task_type, second, second_box, 2, 2), "bounding_box": second_box.model_dump()}, depends_on=[1], bounding_boxes=[second_box]),
            ])
            return calls
        for position, (image, box) in enumerate(zip(images, boxes), start=1):
            calls.append(ToolCall(name="geochat", purpose="Interpret the supplied image region", operation="region_grounded_analysis", arguments={"image_ref": image.image_id, "prompt": TaskPlanner._region_prompt(prompt, task_type, image, box, position, len(images)), "bounding_box": box.model_dump()}, bounding_boxes=[box]))
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
    def _region_prompt(prompt: str, task_type: TaskType, image: ImageReference, box: BoundingBox, position: int, total: int) -> str:
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
            f"IMAGE {position} OF {total}: {image.image_id}\n"
            f"REGION OF INTEREST: {box.compact()}\n"
            f"INSTRUCTION: {instructions[task_type]} Analyze only the supplied region; do not rely on unrelated image areas."
        )
