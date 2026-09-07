from __future__ import annotations

import json
import logging
from typing import Any

import gradio as gr
import spaces

from qwen.controller.controller import QwenController
from qwen.controller.executor import ToolExecutor
from qwen.controller.model import QwenEngine
from qwen.controller.schemas import (
    AnalysisRequest,
    build_input_manifest,
)
from qwen.controller.services import (
    MAX_CONTROLLER_STEPS,
    validate_config,
)


logger = logging.getLogger("akasha.app")


SPACE_VERSION = "manifest-orchestration-v3"


# ----------------------------------------------------------------------
# Startup validation / controller construction
# ----------------------------------------------------------------------

validate_config()

controller = QwenController(
    qwen=QwenEngine(),
    executor=ToolExecutor(),
    max_steps=MAX_CONTROLLER_STEPS,
)


# ----------------------------------------------------------------------
# Input normalization
# ----------------------------------------------------------------------

def _normalize_physical_files(
    physical_files: Any,
) -> list[str]:
    """
    Convert Gradio file values into local filesystem references.

    Gradio may provide:
        - a single filepath string
        - a list of filepath strings
        - file-like dictionaries containing `path` / `name`

    No filenames are interpreted for scientific metadata here.
    """

    if not physical_files:
        return []

    files = (
        physical_files
        if isinstance(physical_files, list)
        else [physical_files]
    )

    result: list[str] = []

    for item in files:
        if not item:
            continue

        if isinstance(item, str):
            path = item.strip()

        elif isinstance(item, dict):
            path = str(
                item.get("path")
                or item.get("name")
                or ""
            ).strip()

        else:
            path = str(
                getattr(item, "path", "")
                or getattr(item, "name", "")
                or ""
            ).strip()

        if path:
            result.append(path)

    return result


def _normalize_urls(
    values: tuple[str, ...],
) -> list[str]:
    return [
        value.strip()
        for value in values
        if isinstance(value, str)
        and value.strip()
    ]


def _load_manifest_json(
    manifest_json: str,
) -> dict[str, Any]:
    if not manifest_json or not manifest_json.strip():
        return {}

    try:
        value = json.loads(
            manifest_json
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Manifest JSON is invalid: {exc.msg}"
        ) from exc

    if not isinstance(value, dict):
        raise ValueError(
            "Manifest must be a JSON object."
        )

    return value


# ----------------------------------------------------------------------
# Controller execution
# ----------------------------------------------------------------------

def _run_controller(
    request: AnalysisRequest,
) -> dict[str, Any]:
    """
    Execute the controller.

    Kept behind a small boundary so the Gradio handler itself remains
    responsible only for transport/input normalization.
    """

    return controller.run_request(
        request
    )


# ----------------------------------------------------------------------
# Public Gradio API
# ----------------------------------------------------------------------

def analyze(
    user_request: str,
    url_1: str = "",
    url_2: str = "",
    url_3: str = "",
    url_4: str = "",
    manifest_json: str = "",
    physical_files: Any = None,
) -> str:
    """
    Production-compatible Qwen controller endpoint.

    Accepted physical inputs:
        - up to four HTTPS image references
        - up to four local physical files

    The backend is responsible for authorization and signed URLs in
    production. Direct local files exist primarily for Space testing.
    """

    try:
        if (
            not isinstance(
                user_request,
                str,
            )
            or not user_request.strip()
        ):
            raise ValueError(
                "User Request must not be empty."
            )

        signed_urls = _normalize_urls(
            (
                url_1,
                url_2,
                url_3,
                url_4,
            )
        )

        local_paths = _normalize_physical_files(
            physical_files
        )

        all_refs = [
            *signed_urls,
            *local_paths,
        ]

        if not all_refs:
            raise ValueError(
                "At least one image input is required."
            )

        if len(all_refs) > 4:
            raise ValueError(
                "At most four physical image files are supported."
            )

        # Production requests should use HTTPS signed references.
        # Local files are allowed for direct HF Space testing.
        if any(
            value.startswith("http://")
            for value in signed_urls
        ):
            raise ValueError(
                "Remote image references must use HTTPS."
            )

        metadata = _load_manifest_json(
            manifest_json
        )

        manifest = build_input_manifest(
            all_refs,
            raw_manifest=metadata,
        )

        request = AnalysisRequest(
            user_request=user_request.strip(),
            signed_image_urls=signed_urls,
            local_image_paths=local_paths,
            manifest=manifest,
            metadata=metadata,
        )

        response = _run_controller(
            request
        )

        # Defensive normalization so the public endpoint always returns
        # a JSON-serializable object.
        if not isinstance(
            response,
            dict,
        ):
            response = {
                "status": "completed",
                "answer": str(response),
            }
        else:
            response = dict(
                response
            )

        response["space_version"] = SPACE_VERSION

        return json.dumps(
            response,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    except ValueError as exc:
        logger.warning(
            "Invalid analysis request: %s",
            exc,
        )
        raise gr.Error(
            str(exc)
        ) from exc

    except Exception as exc:
        logger.exception(
            "Qwen controller request failed."
        )
        raise gr.Error(
            "Qwen could not complete the requested analysis."
        ) from exc


# ----------------------------------------------------------------------
# Gradio interface
# ----------------------------------------------------------------------

demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.Textbox(
            label="User Request",
            lines=4,
        ),

        gr.Textbox(
            label="Signed Image URL 1",
        ),

        gr.Textbox(
            label="Signed Image URL 2",
        ),

        gr.Textbox(
            label="Signed Image URL 3",
        ),

        gr.Textbox(
            label="Signed Image URL 4",
        ),

        # Backend transport field.
        # Hidden from ordinary public users, but preserved in the API
        # contract for manifest-aware production calls.
        gr.Textbox(
            visible=False,
        ),

        gr.File(
            label="Direct physical files (up to 4)",
            file_count="multiple",
            type="filepath",
        ),
    ],
    outputs=gr.Code(
        label="Qwen Response",
        language="json",
    ),
    title="SatQuery AI Qwen Controller",
    api_name="analyze",
)


if __name__ == "__main__":
    demo.launch()