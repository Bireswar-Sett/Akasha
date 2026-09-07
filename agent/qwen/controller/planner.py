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
    """Raised when a request cannot be converted into a safe execution plan."""


class TaskPlanner:
    """
    Convert a normalized AnalysisRequest into the smallest safe tool plan.

    Important boundaries:
    - The planner decides WHAT should run.
    - The executor decides HOW it actually runs.
    - Tool names and operations are always explicit.
    - Capability checks are deployment-driven, never inferred merely from
      the number of uploaded files.
    - Logical observations are the planning unit. Physical files are only
      referenced when a specialist adapter needs them.
    """

    def plan(self, request: AnalysisRequest) -> ToolPlan:
        observations = self._observations(request)
        task_type = self._task_type(request.user_request)

        if not observations:
            return ToolPlan(
                task_type=task_type,
                task_description="No logical image observations were supplied.",
                input_configuration=InputConfiguration.SINGLE_IMAGE,
                images_used=[],
                calls=[],
                bounding_boxes=[],
                compatibility_issue={
                    "status": "input_incompatible",
                    "reason": "At least one logical image observation is required.",
                },
            )

        boxes, box_issue = self._regions(
            request,
            len(observations),
        )

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

        configuration = self._configuration(
            request,
            observations,
        )

        if configuration is None:
            return ToolPlan(
                task_type=task_type,
                task_description="The supplied observations do not match a supported input configuration.",
                input_configuration=InputConfiguration.SINGLE_IMAGE,
                images_used=[observation.id for observation in observations],
                calls=[],
                bounding_boxes=boxes,
                compatibility_issue={
                    "status": "input_incompatible",
                    "reason": (
                        "The observations do not form a supported single-image, "
                        "optical+SAR, bi-temporal, or explicitly enabled dual-SAR configuration."
                    ),
                    "logical_observations": len(observations),
                },
            )

        # A temporal request cannot be meaningfully answered from one
        # observation. Do this before tool selection so we never manufacture
        # a temporal specialist call from insufficient input.
        if (
            len(observations) == 1
            and task_type
            in {
                TaskType.CHANGE_ANALYSIS,
                TaskType.TEMPORAL_SEMANTIC,
            }
        ):
            return ToolPlan(
                task_type=task_type,
                task_description="Temporal comparison requires two corresponding image observations.",
                input_configuration=InputConfiguration.SINGLE_IMAGE,
                images_used=[observation.id for observation in observations],
                calls=[],
                bounding_boxes=boxes,
                compatibility_issue={
                    "status": "input_incompatible",
                    "reason": "Temporal comparison requires two corresponding image observations.",
                    "required": "two corresponding image observations",
                    "available": len(observations),
                },
            )

        calls = self._calls(
            request=request,
            observations=observations,
            configuration=configuration,
            task_type=task_type,
            boxes=boxes,
        )

        # Explicit capability failure is preferable to hallucinating that
        # a specialist executed successfully.
        if configuration == InputConfiguration.DUAL_SAR and not calls:
            return ToolPlan(
                task_type=task_type,
                task_description=(
                    "Two SAR observations were normalized, but no deployed "
                    "SAR/SAR temporal specialist is available."
                ),
                input_configuration=configuration,
                images_used=[observation.id for observation in observations],
                calls=[],
                bounding_boxes=boxes,
                compatibility_issue={
                    "status": "capability_unavailable",
                    "reason": (
                        "No deployed specialist explicitly declares support "
                        "for SAR/SAR temporal analysis."
                    ),
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

    # ------------------------------------------------------------------
    # Input / region normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _regions(
        request: AnalysisRequest,
        image_count: int,
    ) -> tuple[list[BoundingBox], dict[str, Any] | None]:
        boxes = list(request.bounding_boxes)

        metadata_boxes = request.metadata.get("bounding_boxes")

        if not boxes and isinstance(metadata_boxes, list):
            try:
                boxes = [
                    BoundingBox.model_validate(item)
                    for item in metadata_boxes
                ]
            except ValueError as exc:
                return [], {
                    "status": "input_incompatible",
                    "reason": str(exc),
                }

        if not boxes:
            return [], None

        # One logical image must not silently receive several unrelated
        # regions. Humans may be comfortable with ambiguity; executors are not.
        if image_count == 1 and len(boxes) != 1:
            return boxes, {
                "status": "input_incompatible",
                "reason": "one logical observation requires exactly one bounding box",
            }

        pair_metadata = request.metadata.get("pair_metadata")
        if not isinstance(pair_metadata, dict):
            pair_metadata = {}

        if (
            image_count > 1
            and len(boxes) == 1
            and pair_metadata.get("same_region") is True
        ):
            return boxes * image_count, None

        if len(boxes) != image_count:
            return boxes, {
                "status": "input_incompatible",
                "reason": (
                    "multiple logical observations require one explicitly "
                    "associated bounding box per observation"
                ),
            }

        return boxes, None

    @staticmethod
    def _observations(request: AnalysisRequest) -> list[Observation]:
        return list(request.manifest.observations)

    @staticmethod
    def _images(request: AnalysisRequest) -> list[ImageReference]:
        """
        Return physical references for compatibility with older callers.

        Planning itself works on logical observations.
        """
        return [
            image
            for observation in request.manifest.observations
            for image in observation.physical_files
        ]

    # ------------------------------------------------------------------
    # Task / configuration classification
    # ------------------------------------------------------------------

    @staticmethod
    def _task_type(text: str) -> TaskType:
        """
        Deterministic intent classifier.

        Explicit metadata intent wins over text classification when the caller
        already normalized it upstream. The textual fallback exists for
        backwards compatibility with the current request contract.
        """
        value = (text or "").strip().lower()

        # Most specific temporal/change intents first.
        if re.search(
            r"\b(change|changed|difference|before and after|temporal|between)\b",
            value,
        ):
            if re.search(
                r"\b(what|which|why|how|is|are|did|does)\b",
                value,
            ):
                return TaskType.TEMPORAL_SEMANTIC

            return TaskType.CHANGE_ANALYSIS

        if re.search(
            r"\b(where|locate|location|highlight|outline|region|area)\b",
            value,
        ):
            return TaskType.REGION_GROUNDING

        if re.search(
            r"\b(describe|description|caption|scene|landscape)\b",
            value,
        ):
            return TaskType.SCENE_DESCRIPTION

        if re.search(
            r"\b(identify|extract|classify|classification|land[- ]cover)\b",
            value,
        ):
            return TaskType.INFORMATION_EXTRACTION

        return TaskType.VISUAL_QA

    @staticmethod
    def _configuration(
        request: AnalysisRequest,
        observations: list[Observation],
    ) -> InputConfiguration | None:
        count = len(observations)

        if count == 1:
            return InputConfiguration.SINGLE_IMAGE

        if count != 2:
            return None

        explicit = request.metadata.get("input_configuration")

        aliases = {
            "optical_sar": InputConfiguration.OPTICAL_SAR,
            "cross_modal": InputConfiguration.OPTICAL_SAR,
            "bi_temporal": InputConfiguration.BI_TEMPORAL,
            "temporal": InputConfiguration.BI_TEMPORAL,
            "dual_sar": InputConfiguration.DUAL_SAR,
        }

        if isinstance(explicit, InputConfiguration):
            return explicit

        if isinstance(explicit, str) and explicit in aliases:
            return aliases[explicit]

        modalities = {
            observation.modality
            for observation in observations
        }

        # Cross-modal optical + SAR.
        if modalities in (
            {"optical", "sar"},
            {"multispectral", "sar"},
        ):
            return InputConfiguration.OPTICAL_SAR

        # SAR + SAR is its own configuration because two SAR observations
        # contain VV/VH pairs internally.
        if all(
            observation.modality == "sar"
            for observation in observations
        ):
            if all(
                observation.acquisition_time
                for observation in observations
            ):
                return InputConfiguration.DUAL_SAR

            return None

        # Remaining two-image configuration is temporal only when the
        # relationship metadata confirms correspondence.
        pair_metadata = request.metadata.get("pair_metadata")
        if not isinstance(pair_metadata, dict):
            pair_metadata = {}

        spatially_corresponding = pair_metadata.get(
            "spatially_corresponding"
        )

        if spatially_corresponding is None:
            spatially_corresponding = (
                request.manifest.relationship.spatially_corresponding
            )

        if (
            all(observation.acquisition_time for observation in observations)
            and spatially_corresponding is True
        ):
            return InputConfiguration.BI_TEMPORAL

        return None

    # ------------------------------------------------------------------
    # Tool planning
    # ------------------------------------------------------------------

    @staticmethod
    def _calls(
        request: AnalysisRequest,
        observations: list[Observation],
        configuration: InputConfiguration,
        task_type: TaskType,
        boxes: list[BoundingBox],
    ) -> list[ToolCall]:
        prompt = request.user_query or request.user_request

        if boxes:
            return TaskPlanner._region_calls(
                request=request,
                prompt=prompt,
                observations=observations,
                configuration=configuration,
                task_type=task_type,
                boxes=boxes,
            )

        prompt = TaskPlanner._standard_prompt(
            prompt,
            task_type,
        )

        # --------------------------------------------------------------
        # Single observation
        # --------------------------------------------------------------

        if configuration == InputConfiguration.SINGLE_IMAGE:
            observation = observations[0]

            # SAR is physically VV + VH but must become one logical visual
            # artifact before GeoChat sees it.
            if observation.modality == "sar":
                return [
                    ToolCall(
                        name="pseudo_rgb",
                        purpose="Prepare the SAR observation for semantic vision.",
                        operation="sar_to_pseudo_rgb",
                        arguments={
                            "observation_id": observation.id,
                        },
                    ),
                    ToolCall(
                        name="geochat",
                        purpose="Interpret the requested SAR image.",
                        operation="single_image_analysis",
                        arguments={
                            "image_ref": "artifact:pseudo_rgb",
                            "prompt": prompt,
                        },
                        depends_on=[1],
                    ),
                ]

            return [
                ToolCall(
                    name="geochat",
                    purpose="Interpret the requested optical image.",
                    operation="single_image_analysis",
                    arguments={
                        "observation_id": observation.id,
                        "prompt": prompt,
                    },
                )
            ]

        # --------------------------------------------------------------
        # Optical + SAR
        # --------------------------------------------------------------

        if configuration == InputConfiguration.OPTICAL_SAR:
            optical = next(
                observation
                for observation in observations
                if observation.modality in {"optical", "multispectral"}
            )

            sar = next(
                observation
                for observation in observations
                if observation.modality == "sar"
            )

            # M²CD should only be used here when the deployed service
            # explicitly supports optical+SAR change detection.
            if task_type in {
                TaskType.CHANGE_ANALYSIS,
                TaskType.TEMPORAL_SEMANTIC,
            }:
                if TaskPlanner._capability(
                    request,
                    "m2cd_optical_sar",
                    False,
                ) is True:
                    return [
                        ToolCall(
                            name="m2cd",
                            purpose=(
                                "Detect optical-SAR change evidence "
                                "using the explicitly enabled M²CD capability."
                            ),
                            operation="change_detection",
                            arguments={
                                "observation_1_id": optical.id,
                                "observation_2_id": sar.id,
                            },
                        )
                    ]

                # No supported change detector means no invented change
                # result. We can still provide semantic evidence from both
                # modalities, but only if the planner chooses to do so.
                return [
                    ToolCall(
                        name="geochat",
                        purpose="Interpret optical evidence.",
                        operation="single_image_analysis",
                        arguments={
                            "observation_id": optical.id,
                            "prompt": prompt,
                        },
                    ),
                    ToolCall(
                        name="pseudo_rgb",
                        purpose="Prepare SAR evidence for semantic vision.",
                        operation="sar_to_pseudo_rgb",
                        arguments={
                            "observation_id": sar.id,
                        },
                    ),
                    ToolCall(
                        name="geochat",
                        purpose="Interpret SAR evidence.",
                        operation="single_image_analysis",
                        arguments={
                            "image_ref": "artifact:pseudo_rgb",
                            "prompt": prompt,
                        },
                        depends_on=[2],
                    ),
                ]

            # Non-change optical+SAR analysis.
            return [
                ToolCall(
                    name="geochat",
                    purpose="Interpret optical evidence.",
                    operation="single_image_analysis",
                    arguments={
                        "observation_id": optical.id,
                        "prompt": prompt,
                    },
                ),
                ToolCall(
                    name="pseudo_rgb",
                    purpose="Prepare SAR evidence for semantic vision.",
                    operation="sar_to_pseudo_rgb",
                    arguments={
                        "observation_id": sar.id,
                    },
                ),
                ToolCall(
                    name="geochat",
                    purpose="Interpret SAR evidence.",
                    operation="single_image_analysis",
                    arguments={
                        "image_ref": "artifact:pseudo_rgb",
                        "prompt": prompt,
                    },
                    depends_on=[2],
                ),
            ]

        # --------------------------------------------------------------
        # Two corresponding observations
        # --------------------------------------------------------------

        if configuration == InputConfiguration.BI_TEMPORAL:
            return [
                ToolCall(
                    name="teochat",
                    purpose="Perform temporal reasoning over corresponding observations.",
                    operation="temporal_analysis",
                    arguments={
                        "observation_1_id": observations[0].id,
                        "observation_2_id": observations[1].id,
                        "prompt": prompt,
                    },
                )
            ]

        # --------------------------------------------------------------
        # Two SAR observations
        # --------------------------------------------------------------

        if configuration == InputConfiguration.DUAL_SAR:
            if TaskPlanner._capability(
                request,
                "m2cd_sar_sar",
                False,
            ) is True:
                return [
                    ToolCall(
                        name="m2cd",
                        purpose=(
                            "Detect change between two SAR observations "
                            "using the explicitly enabled SAR/SAR capability."
                        ),
                        operation="change_detection",
                        arguments={
                            "observation_1_id": observations[0].id,
                            "observation_2_id": observations[1].id,
                        },
                    )
                ]

            return []

        return []

    # ------------------------------------------------------------------
    # Region-aware planning
    # ------------------------------------------------------------------

    @staticmethod
    def _region_calls(
        request: AnalysisRequest,
        prompt: str,
        observations: list[Observation],
        configuration: InputConfiguration,
        task_type: TaskType,
        boxes: list[BoundingBox],
    ) -> list[ToolCall]:
        calls: list[ToolCall] = []

        # --------------------------------------------------------------
        # Temporal / dual-SAR region analysis
        # --------------------------------------------------------------

        if configuration in {
            InputConfiguration.BI_TEMPORAL,
            InputConfiguration.DUAL_SAR,
        }:
            first = observations[0]
            second = observations[1]

            first_box = boxes[0]
            second_box = boxes[1]

            # M²CD is only valid when the actual deployed capability says
            # that this configuration is supported.
            if configuration == InputConfiguration.BI_TEMPORAL:
                change_tool = "teochat"
                change_operation = "temporal_analysis"
            else:
                if TaskPlanner._capability(
                    request,
                    "m2cd_sar_sar",
                    False,
                ) is not True:
                    return []

                change_tool = "m2cd"
                change_operation = "change_detection"

            if change_tool == "m2cd":
                calls.append(
                    ToolCall(
                        name="m2cd",
                        purpose="Detect temporal change for the requested regions.",
                        operation=change_operation,
                        arguments={
                            "observation_1_id": first.id,
                            "observation_2_id": second.id,
                        },
                        bounding_boxes=[
                            first_box,
                            second_box,
                        ],
                    )
                )

                depends_on = [1]
            else:
                calls.append(
                    ToolCall(
                        name="teochat",
                        purpose="Perform temporal reasoning for the requested regions.",
                        operation=change_operation,
                        arguments={
                            "observation_1_id": first.id,
                            "observation_2_id": second.id,
                            "prompt": prompt,
                        },
                        bounding_boxes=[
                            first_box,
                            second_box,
                        ],
                    )
                )

                depends_on = [1]

            calls.extend(
                [
                    ToolCall(
                        name="geochat",
                        purpose="Interpret the requested T1 region.",
                        operation="region_grounded_analysis",
                        arguments={
                            "observation_id": first.id,
                            "prompt": TaskPlanner._region_prompt(
                                prompt,
                                task_type,
                                first,
                                first_box,
                                1,
                                2,
                            ),
                            "bounding_box": first_box.model_dump(),
                        },
                        depends_on=depends_on,
                        bounding_boxes=[first_box],
                    ),
                    ToolCall(
                        name="geochat",
                        purpose="Interpret the requested T2 region.",
                        operation="region_grounded_analysis",
                        arguments={
                            "observation_id": second.id,
                            "prompt": TaskPlanner._region_prompt(
                                prompt,
                                task_type,
                                second,
                                second_box,
                                2,
                                2,
                            ),
                            "bounding_box": second_box.model_dump(),
                        },
                        depends_on=depends_on,
                        bounding_boxes=[second_box],
                    ),
                ]
            )

            return calls

        # --------------------------------------------------------------
        # Optical + SAR regional analysis
        # --------------------------------------------------------------

        if configuration == InputConfiguration.OPTICAL_SAR:
            optical = next(
                observation
                for observation in observations
                if observation.modality in {"optical", "multispectral"}
            )

            sar = next(
                observation
                for observation in observations
                if observation.modality == "sar"
            )

            optical_box = boxes[0]
            sar_box = boxes[1]

            if task_type in {
                TaskType.CHANGE_ANALYSIS,
                TaskType.TEMPORAL_SEMANTIC,
            }:
                if TaskPlanner._capability(
                    request,
                    "m2cd_optical_sar",
                    False,
                ) is True:
                    return [
                        ToolCall(
                            name="m2cd",
                            purpose=(
                                "Detect optical-SAR change in the requested regions."
                            ),
                            operation="change_detection",
                            arguments={
                                "observation_1_id": optical.id,
                                "observation_2_id": sar.id,
                            },
                            bounding_boxes=[
                                optical_box,
                                sar_box,
                            ],
                        ),
                    ]

            # Semantic interpretation of each modality independently.
            calls.extend(
                [
                    ToolCall(
                        name="geochat",
                        purpose="Interpret the requested optical region.",
                        operation="region_grounded_analysis",
                        arguments={
                            "observation_id": optical.id,
                            "prompt": TaskPlanner._region_prompt(
                                prompt,
                                task_type,
                                optical,
                                optical_box,
                                1,
                                2,
                            ),
                            "bounding_box": optical_box.model_dump(),
                        },
                        bounding_boxes=[optical_box],
                    ),
                    ToolCall(
                        name="pseudo_rgb",
                        purpose="Prepare the requested SAR region for semantic vision.",
                        operation="sar_to_pseudo_rgb",
                        arguments={
                            "observation_id": sar.id,
                        },
                        bounding_boxes=[sar_box],
                    ),
                    ToolCall(
                        name="geochat",
                        purpose="Interpret the requested SAR region.",
                        operation="region_grounded_analysis",
                        arguments={
                            "image_ref": "artifact:pseudo_rgb",
                            "prompt": TaskPlanner._region_prompt(
                                prompt,
                                task_type,
                                sar,
                                sar_box,
                                2,
                                2,
                            ),
                            "bounding_box": sar_box.model_dump(),
                        },
                        depends_on=[2],
                        bounding_boxes=[sar_box],
                    ),
                ]
            )

            return calls

        # --------------------------------------------------------------
        # Single-image / generic regional analysis
        # --------------------------------------------------------------

        for position, (observation, box) in enumerate(
            zip(observations, boxes),
            start=1,
        ):
            if observation.modality == "sar":
                calls.append(
                    ToolCall(
                        name="pseudo_rgb",
                        purpose="Prepare the SAR region for semantic vision.",
                        operation="sar_to_pseudo_rgb",
                        arguments={
                            "observation_id": observation.id,
                        },
                        bounding_boxes=[box],
                    )
                )

                calls.append(
                    ToolCall(
                        name="geochat",
                        purpose="Interpret the requested SAR region.",
                        operation="region_grounded_analysis",
                        arguments={
                            "image_ref": "artifact:pseudo_rgb",
                            "prompt": TaskPlanner._region_prompt(
                                prompt,
                                task_type,
                                observation,
                                box,
                                position,
                                len(observations),
                            ),
                            "bounding_box": box.model_dump(),
                        },
                        depends_on=[len(calls)],
                        bounding_boxes=[box],
                    )
                )
            else:
                calls.append(
                    ToolCall(
                        name="geochat",
                        purpose="Interpret the supplied image region.",
                        operation="region_grounded_analysis",
                        arguments={
                            "observation_id": observation.id,
                            "prompt": TaskPlanner._region_prompt(
                                prompt,
                                task_type,
                                observation,
                                box,
                                position,
                                len(observations),
                            ),
                            "bounding_box": box.model_dump(),
                        },
                        bounding_boxes=[box],
                    )
                )

        return calls

    # ------------------------------------------------------------------
    # Capability / prompts
    # ------------------------------------------------------------------

    @staticmethod
    def _capability(
        request: AnalysisRequest,
        name: str,
        default: Any,
    ) -> Any:
        capabilities = request.metadata.get("capabilities")

        if not isinstance(capabilities, dict):
            capabilities = request.manifest.metadata.get("capabilities")

        if not isinstance(capabilities, dict):
            return default

        return capabilities.get(name, default)

    @staticmethod
    def _description(task_type: TaskType) -> str:
        return {
            TaskType.VISUAL_QA:
                "Answer a visual question from specialist image evidence.",
            TaskType.SCENE_DESCRIPTION:
                "Describe the scene using specialist visual evidence.",
            TaskType.REGION_GROUNDING:
                "Locate the requested content and preserve supported spatial evidence.",
            TaskType.INFORMATION_EXTRACTION:
                "Extract visually supported remote-sensing information.",
            TaskType.CHANGE_ANALYSIS:
                "Determine and describe temporal change evidence.",
            TaskType.TEMPORAL_SEMANTIC:
                "Answer a semantic question about temporal change.",
        }[task_type]

    @staticmethod
    def _standard_prompt(
        prompt: str,
        task_type: TaskType,
    ) -> str:
        instructions = {
            TaskType.VISUAL_QA:
                "Answer the visual question using evidence from the image.",
            TaskType.SCENE_DESCRIPTION:
                "Describe the scene and its major visible land-cover types and features.",
            TaskType.REGION_GROUNDING:
                "Identify and localize the requested feature, preserving supported spatial evidence.",
            TaskType.INFORMATION_EXTRACTION:
                "Extract the requested remote-sensing information from visible evidence.",
            TaskType.CHANGE_ANALYSIS:
                "Analyze the relevant image evidence for the requested temporal difference.",
            TaskType.TEMPORAL_SEMANTIC:
                "Interpret the requested temporal relationship using evidence from the supplied imagery.",
        }

        return (
            f"USER TASK: {prompt}\n"
            f"INSTRUCTION: {instructions[task_type]}\n"
            "Report observations separately from interpretation and uncertainty. "
            "Do not infer facts that are not supported by the supplied imagery."
        )

    @staticmethod
    def _region_prompt(
        prompt: str,
        task_type: TaskType,
        image: Observation,
        box: BoundingBox,
        position: int,
        total: int,
    ) -> str:
        instructions = {
            TaskType.VISUAL_QA:
                "Answer the visual question using evidence specifically from this region.",
            TaskType.SCENE_DESCRIPTION:
                "Describe this region's visible land-cover and major features.",
            TaskType.REGION_GROUNDING:
                "Identify the requested feature within this region and provide supported spatial evidence.",
            TaskType.INFORMATION_EXTRACTION:
                "Extract the requested information from this region only.",
            TaskType.CHANGE_ANALYSIS:
                "Describe temporal differences supported specifically by this region.",
            TaskType.TEMPORAL_SEMANTIC:
                "Interpret the requested temporal relationship using evidence from this region.",
        }

        return (
            f"USER TASK: {prompt}\n"
            f"OBSERVATION {position} OF {total}: {image.id}\n"
            f"REGION OF INTEREST: {box.compact()}\n"
            f"INSTRUCTION: {instructions[task_type]}\n"
            "Analyze only the supplied region; do not rely on unrelated image areas."
        )