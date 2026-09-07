from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from qwen.controller.executor import ToolExecutor
from qwen.controller.model import QwenEngine
from qwen.controller.planner import PlanningError, TaskPlanner
from qwen.controller.prompts import FINAL_REASONING_PROMPT
from qwen.controller.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    ExecutionTrace,
    ImageReference,
    InputManifest,
    Observation,
    RelationshipMetadata,
    build_input_manifest,
    SpecialistEvidence,
    ToolStatus,
)

logger = logging.getLogger("satquery.controller")


class QwenController:
    """Plan, execute, and synthesize one isolated analysis request."""

    def __init__(
        self,
        qwen: QwenEngine,
        executor: ToolExecutor,
        max_steps: int = 8,
        planner: TaskPlanner | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")

        self.qwen = qwen
        self.executor = executor
        self.max_steps = max_steps
        self.planner = planner or TaskPlanner()

    def run_request(
        self,
        request: AnalysisRequest,
        max_new_tokens: int = 512,
    ) -> dict[str, Any]:
        try:
            plan = self.planner.plan(request)
        except (PlanningError, ValueError) as exc:
            return AnalysisResponse(
                status="input_incompatible",
                error={"reason": str(exc)},
            ).model_dump()

        base = {
            "task": {
                "type": plan.task_type.value,
                "description": plan.task_description,
            },
            "input": {
                "configuration": plan.input_configuration.value,
                "images_used": plan.images_used,
                "user_query": request.user_query,
                "bounding_boxes": [
                    box.model_dump()
                    for box in plan.bounding_boxes
                ],
            },
        }

        if plan.compatibility_issue:
            return AnalysisResponse(
                status="input_incompatible",
                **base,
                error=plan.compatibility_issue,
            ).model_dump()

        images = self.planner._images(request)

        image_refs = {
            image.image_id: image.url
            for image in images
        }

        evidence: list[SpecialistEvidence] = []
        trace: list[ExecutionTrace] = []
        artifacts: dict[str, Any] = {}

        # IMPORTANT:
        # Dependency indices are 0-based and correspond directly
        # to plan.calls.
        #
        #   plan.calls[0] -> first tool
        #   plan.calls[1] -> second tool
        #   ...
        #
        # The execution trace displayed externally remains 1-based.
        step_status: dict[int, str] = {}

        for call_index, call in enumerate(plan.calls):

            # ---------------------------------------------------------
            # Dependency resolution
            # ---------------------------------------------------------
            #
            # call.depends_on contains indices into plan.calls.
            # Therefore it MUST use the same 0-based indexing.
            #
            # Example:
            #
            #   calls[0] = pseudo_rgb
            #   calls[1] = geochat
            #
            #   geochat.depends_on = [0]
            #
            blocked_by = [
                dependency
                for dependency in call.depends_on
                if step_status.get(dependency)
                != ToolStatus.COMPLETED.value
            ]

            # Human-facing execution step numbers remain 1-based.
            trace_step = call_index + 1

            # ---------------------------------------------------------
            # Dependency failure
            # ---------------------------------------------------------
            if blocked_by:
                step_status[call_index] = ToolStatus.SKIPPED.value

                trace.append(
                    ExecutionTrace(
                        step=trace_step,
                        tool=call.name,
                        operation=call.operation,
                        status=ToolStatus.SKIPPED,
                        details={
                            "depends_on": blocked_by
                        },
                        bounding_boxes=call.bounding_boxes,
                    )
                )

                evidence.append(
                    SpecialistEvidence(
                        tool=call.name,
                        operation=call.operation,
                        success=False,
                        status=ToolStatus.SKIPPED,
                        error_summary=(
                            "Skipped because a dependent tool "
                            "did not complete."
                        ),
                        bounding_boxes=call.bounding_boxes,
                    )
                )

                continue

            # ---------------------------------------------------------
            # Resolve authorized image/artifact references
            # ---------------------------------------------------------
            arguments = self._bind_authorized_references(
                call.arguments,
                image_refs,
                artifacts,
                request.manifest,
                call.name,
            )

            # ---------------------------------------------------------
            # Execute specialist
            # ---------------------------------------------------------
            started = perf_counter()

            result = self.executor.execute(
                call.name,
                arguments,
            )

            duration_ms = int(
                (perf_counter() - started) * 1000
            )

            ok = bool(result.get("ok"))

            result_status = (
                ToolStatus.COMPLETED.value
                if ok
                else str(
                    result.get("status")
                    or ToolStatus.FAILED.value
                )
            )

            # IMPORTANT:
            # Store status using call_index, NOT trace_step.
            #
            # This keeps dependency resolution 0-based.
            step_status[call_index] = result_status

            trace.append(
                ExecutionTrace(
                    step=trace_step,
                    tool=call.name,
                    operation=call.operation,
                    status=result_status,
                    duration_ms=duration_ms,
                    bounding_boxes=call.bounding_boxes,
                )
            )

            # ---------------------------------------------------------
            # Specialist failure
            # ---------------------------------------------------------
            if not ok:
                evidence.append(
                    SpecialistEvidence(
                        tool=call.name,
                        operation=call.operation,
                        success=False,
                        status=result_status,
                        error_summary=str(
                            result.get(
                                "error",
                                "specialist failed",
                            )
                        ),
                        bounding_boxes=call.bounding_boxes,
                    )
                )

                continue

            # ---------------------------------------------------------
            # Successful specialist result
            # ---------------------------------------------------------
            result_value = result.get(
                "result",
                result,
            )

            # ---------------------------------------------------------
            # pseudo_rgb artifact registration
            # ---------------------------------------------------------
            #
            # GeoChat can subsequently reference:
            #
            #     artifact:pseudo_rgb
            #
            # and this gets resolved to the actual image reference.
            #
            if (
                call.name == "pseudo_rgb"
                and isinstance(result_value, dict)
            ):
                artifacts["artifact:pseudo_rgb"] = (
                    result_value.get("image_ref")
                )

            # ---------------------------------------------------------
            # Record specialist evidence
            # ---------------------------------------------------------
            evidence.append(
                SpecialistEvidence(
                    tool=call.name,
                    operation=call.operation,
                    result=result_value,
                    bounding_boxes=call.bounding_boxes,
                    success=True,
                    status=ToolStatus.COMPLETED,
                )
            )

        # -------------------------------------------------------------
        # Final Qwen synthesis
        # -------------------------------------------------------------
        answer = self._synthesize(
            request.user_query
            or request.user_request,
            base,
            evidence,
            max_new_tokens,
        )

        has_failures = any(
            item.status
            not in {
                ToolStatus.COMPLETED,
                ToolStatus.AVAILABLE,
            }
            for item in evidence
        )

        return AnalysisResponse(
            status=(
                "completed_with_warnings"
                if has_failures
                else "completed"
            ),
            **base,
            execution=trace,
            evidence=evidence,
            answer=answer,
        ).model_dump()

    def run(
        self,
        user_message: str,
        manifest: dict[str, Any],
        image_refs: dict[str, str | None],
        max_new_tokens: int = 512,
    ) -> dict[str, Any]:

        urls = [
            value
            for value in image_refs.values()
            if value
        ]

        request_manifest = self._authorized_manifest(
            manifest,
            image_refs,
        )

        request = AnalysisRequest(
            user_request=user_message,
            signed_image_urls=urls,
            manifest=request_manifest,
            metadata=manifest,
        )

        return self.run_request(
            request,
            max_new_tokens,
        )

    @staticmethod
    def _authorized_manifest(
        manifest: dict[str, Any],
        image_refs: dict[str, str | None],
    ) -> InputManifest:

        return build_input_manifest(
            [
                value
                for value in image_refs.values()
                if value
            ],
            raw_manifest=manifest,
            authorized_refs=image_refs,
        )

    def _synthesize(
        self,
        user_request: str,
        base: dict[str, Any],
        evidence: list[SpecialistEvidence],
        max_new_tokens: int,
    ) -> str:

        messages = [
            {
                "role": "system",
                "content": FINAL_REASONING_PROMPT,
            },
            {
                "role": "user",
                "content": self._final_reasoning_message(
                    user_request,
                    base,
                    [
                        item.model_dump()
                        for item in evidence
                    ],
                ),
            },
        ]

        answer = self.qwen.chat(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            do_sample=False,
        )

        if not str(answer).strip():
            raise RuntimeError(
                "Qwen produced an empty response"
            )

        return str(answer).strip()

    @staticmethod
    def _final_reasoning_message(
        user_request: str,
        base: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> str:

        import json

        return (
            "USER REQUEST\n"
            + user_request
            + "\n\nINPUT\n"
            + json.dumps(base, default=str)
            + "\n\nSPECIALIST EVIDENCE\n"
            + json.dumps(evidence, default=str)
            + "\n\nAnswer only from this evidence; "
            "do not expose hidden reasoning."
        )

    @staticmethod
    def _bind_authorized_references(
        arguments: dict[str, Any],
        image_refs: dict[str, str],
        artifacts: dict[str, Any],
        manifest: InputManifest,
        tool_name: str,
    ) -> dict[str, Any]:

        bound = dict(arguments)

        observations = {
            observation.id: observation
            for observation in manifest.observations
        }

        def observation_files(
            observation_id: str,
        ) -> tuple[str, ...]:

            observation = observations.get(
                observation_id
            )

            if observation is None:
                raise ValueError(
                    "Unknown logical observation ID: "
                    f"{observation_id}"
                )

            return tuple(
                image_refs[image.image_id]
                for image in observation.physical_files
            )

        # -------------------------------------------------------------
        # Single observation
        # -------------------------------------------------------------
        if isinstance(
            bound.get("observation_id"),
            str,
        ):

            refs = observation_files(
                bound.pop("observation_id")
            )

            if tool_name == "pseudo_rgb":
                bound["vv_ref"] = refs[0]
                bound["vh_ref"] = refs[1]
            else:
                bound["image_ref"] = refs[0]

        # -------------------------------------------------------------
        # Two observations
        # -------------------------------------------------------------
        for logical_key, target_key in (
            (
                "observation_1_id",
                "image_1_ref",
            ),
            (
                "observation_2_id",
                "image_2_ref",
            ),
        ):

            if isinstance(
                bound.get(logical_key),
                str,
            ):

                refs = observation_files(
                    bound.pop(logical_key)
                )

                if (
                    tool_name == "m2cd"
                    and len(refs) == 2
                ):

                    prefix = (
                        "image_t1"
                        if logical_key
                        == "observation_1_id"
                        else "image_t2"
                    )

                    bound[
                        f"{prefix}_vv_ref"
                    ] = refs[0]

                    bound[
                        f"{prefix}_vh_ref"
                    ] = refs[1]

                elif tool_name == "m2cd":

                    bound[
                        (
                            "image_t1_ref"
                            if logical_key
                            == "observation_1_id"
                            else "image_t2_ref"
                        )
                    ] = refs[0]

                else:

                    bound[target_key] = refs[0]

        # -------------------------------------------------------------
        # Resolve logical image/artifact references
        # -------------------------------------------------------------
        for key, value in list(
            bound.items()
        ):

            if (
                not key.endswith("ref")
                or not isinstance(value, str)
            ):
                continue

            # Logical image ID → authorized signed URL
            if value in image_refs:
                bound[key] = image_refs[value]

            # Generated artifact → actual artifact reference
            elif value in artifacts:
                bound[key] = artifacts[value]

        return bound
