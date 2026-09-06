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
    SpecialistEvidence,
)

logger = logging.getLogger("satquery.controller")


class QwenController:
    """Plan, execute, and synthesize one isolated analysis request."""

    def __init__(self, qwen: QwenEngine, executor: ToolExecutor, max_steps: int = 8, planner: TaskPlanner | None = None) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.qwen = qwen
        self.executor = executor
        self.max_steps = max_steps
        self.planner = planner or TaskPlanner()

    def run_request(self, request: AnalysisRequest, max_new_tokens: int = 512) -> dict[str, Any]:
        try:
            plan = self.planner.plan(request)
        except (PlanningError, ValueError) as exc:
            return AnalysisResponse(status="input_incompatible", error={"reason": str(exc)}).model_dump()

        base = {
            "task": {"type": plan.task_type.value, "description": plan.task_description},
            "input": {
                "configuration": plan.input_configuration.value,
                "images_used": plan.images_used,
                "user_query": request.user_query,
                "bounding_boxes": [box.model_dump() for box in plan.bounding_boxes],
            },
        }
        if plan.compatibility_issue:
            return AnalysisResponse(status="input_incompatible", **base, error=plan.compatibility_issue).model_dump()

        images = self.planner._images(request)
        image_refs = {image.image_id: image.url for image in images}
        evidence: list[SpecialistEvidence] = []
        trace: list[ExecutionTrace] = []
        artifacts: dict[str, Any] = {}

        for index, call in enumerate(plan.calls, start=1):
            arguments = self._bind_authorized_references(call.arguments, image_refs, artifacts)
            started = perf_counter()
            result = self.executor.execute(call.name, arguments)
            duration_ms = int((perf_counter() - started) * 1000)
            ok = bool(result.get("ok"))
            trace.append(ExecutionTrace(step=index, tool=call.name, operation=call.operation, status="completed" if ok else "failed", duration_ms=duration_ms, bounding_boxes=call.bounding_boxes))
            if not ok:
                return AnalysisResponse(status="failed", **base, execution=trace, evidence=evidence, error={"tool": call.name, "reason": result.get("error", "specialist failed")}).model_dump()
            result_value = result.get("result", result)
            if call.name == "pseudo_rgb" and isinstance(result_value, dict):
                artifacts["artifact:pseudo_rgb"] = result_value.get("image_ref")
            evidence.append(SpecialistEvidence(tool=call.name, operation=call.operation, result=result_value, bounding_boxes=call.bounding_boxes))

        answer = self._synthesize(request.user_query or request.user_request, base, evidence, max_new_tokens)
        return AnalysisResponse(status="completed", **base, execution=trace, evidence=evidence, answer=answer).model_dump()

    def run(self, user_message: str, manifest: dict[str, Any], image_refs: dict[str, str | None], max_new_tokens: int = 512) -> dict[str, Any]:
        urls = [value for value in image_refs.values() if value]
        request = AnalysisRequest(user_request=user_message, signed_image_urls=urls, metadata=manifest)
        return self.run_request(request, max_new_tokens)

    def _synthesize(self, user_request: str, base: dict[str, Any], evidence: list[SpecialistEvidence], max_new_tokens: int) -> str:
        messages = [
            {"role": "system", "content": FINAL_REASONING_PROMPT},
            {"role": "user", "content": self._final_reasoning_message(user_request, base, [item.model_dump() for item in evidence])},
        ]
        answer = self.qwen.chat(messages, max_new_tokens=max_new_tokens, temperature=0.1, top_p=0.9, do_sample=False)
        if not str(answer).strip():
            raise RuntimeError("Qwen produced an empty response")
        return str(answer).strip()

    @staticmethod
    def _final_reasoning_message(user_request: str, base: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
        import json
        return "USER REQUEST\n" + user_request + "\n\nINPUT\n" + json.dumps(base, default=str) + "\n\nSPECIALIST EVIDENCE\n" + json.dumps(evidence, default=str) + "\n\nAnswer only from this evidence; do not expose hidden reasoning."

    @staticmethod
    def _bind_authorized_references(arguments: dict[str, Any], image_refs: dict[str, str], artifacts: dict[str, Any]) -> dict[str, Any]:
        bound = dict(arguments)
        for key, value in list(bound.items()):
            if not key.endswith("ref") or not isinstance(value, str):
                continue
            if value in image_refs:
                bound[key] = image_refs[value]
            elif value in artifacts:
                bound[key] = artifacts[value]
        return bound
