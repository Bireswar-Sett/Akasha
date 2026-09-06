from __future__ import annotations

import json
import logging
from typing import Any

from gradio_client import Client, handle_file

from qwen.controller.image_processing import (
    extract_change_regions,
    load_numeric_mask,
    sar_to_pseudo_rgb_file,
)
from qwen.controller.schemas import BoundingBox
from qwen.controller.services import (
    GEOCHAT_API_NAME,
    GEOCHAT_SPACE,
    HF_TOKEN,
    MAX_CHANGE_REGIONS,
    M2CD_API_NAME,
    M2CD_SPACE,
    M2CD_THRESHOLD,
    TEOCHAT_API_NAME,
    TEOCHAT_SPACE,
)

logger = logging.getLogger("akasha.executor")


class ToolExecutionError(RuntimeError):
    """Raised when a specialist or local processing tool cannot execute."""


def _is_local_file(value: str) -> bool:
    try:
        from pathlib import Path
        return Path(value).is_file()
    except OSError:
        return False


class ToolExecutor:
    """
    The only layer allowed to know how specialist tools are deployed.

    Qwen decides WHAT tool to call.
    This class decides HOW that tool is physically invoked.
    """

    def __init__(
        self,
        hf_token: str | None = None,
    ) -> None:
        self.hf_token = hf_token if hf_token is not None else HF_TOKEN

        if not self.hf_token:
            raise ToolExecutionError(
                "HF_TOKEN is required to call private/specialist Spaces."
            )

        self._clients: dict[str, Client] = {}

    def _client(self, key: str, space: str) -> Client:
        if key not in self._clients:
            logger.info("Initializing HF Space client: %s", space)
            self._clients[key] = Client(
                space,
                token=self.hf_token,
            )
        return self._clients[key]

    @staticmethod
    def _string_arg(
        arguments: dict[str, Any],
        key: str,
    ) -> str:
        value = arguments.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ToolExecutionError(
                f"Tool argument {key!r} must be a non-empty string."
            )
        return value.strip()

    @staticmethod
    def _token_limit(
        arguments: dict[str, Any],
        default: int,
        maximum: int,
    ) -> int:
        value = arguments.get("max_new_tokens", default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ToolExecutionError(
                "max_new_tokens must be an integer."
            )
        if not 1 <= value <= maximum:
            raise ToolExecutionError(
                f"max_new_tokens must be between 1 and {maximum}."
            )
        return value

    @staticmethod
    def _prompt(arguments: dict[str, Any]) -> str:
        return ToolExecutor._string_arg(arguments, "prompt")

    @staticmethod
    def _file_or_url_arg(
        arguments: dict[str, Any],
        key: str,
    ) -> str:
        value = ToolExecutor._string_arg(arguments, key)

        if value.startswith("https://") or value.startswith("http://"):
            return value

        if _is_local_file(value):
            return value

        # Local paths can come from the executor itself. The caller is not
        # allowed to invent arbitrary non-file strings as image references.
        raise ToolExecutionError(
            f"{key!r} must be an HTTPS URL or an existing local file."
        )

    @staticmethod
    def _tool_success(
        tool: str,
        operation: str,
        result: Any,
        **extra: Any,
    ) -> dict[str, Any]:
        return {
            "ok": True,
            "tool": tool,
            "operation": operation,
            "result": result,
            **extra,
        }

    def _execute_geochat(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        image_ref = self._file_or_url_arg(arguments, "image_ref")
        prompt = self._prompt(arguments)
        bounding_box = arguments.get("bounding_box")
        if bounding_box is not None:
            try:
                bounding_box = BoundingBox.model_validate(bounding_box)
            except ValueError as exc:
                raise ToolExecutionError(f"Invalid bounding_box: {exc}") from exc
        max_tokens = self._token_limit(arguments, 256, 512)

        try:
            result = self._client(
                "geochat",
                GEOCHAT_SPACE,
            ).predict(
                image=handle_file(image_ref),
                prompt=prompt,
                max_new_tokens=max_tokens,
                api_name=GEOCHAT_API_NAME,
            )
        except Exception as exc:
            raise ToolExecutionError(
                f"GeoChat request failed: {type(exc).__name__}"
            ) from exc

        return self._tool_success(
            "geochat",
            "region_grounded_analysis" if bounding_box is not None else "single_image_analysis",
            result,
            bounding_box=bounding_box.model_dump() if bounding_box is not None else None,
        )

    def _execute_teochat(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        image_1 = self._file_or_url_arg(arguments, "image_1_ref")
        image_2 = self._file_or_url_arg(arguments, "image_2_ref")
        prompt = self._prompt(arguments)
        max_tokens = self._token_limit(arguments, 384, 768)

        try:
            result = self._client(
                "teochat",
                TEOCHAT_SPACE,
            ).predict(
                image_t1=handle_file(image_1),
                image_t2=handle_file(image_2),
                prompt=prompt,
                max_new_tokens=max_tokens,
                api_name=TEOCHAT_API_NAME,
            )
        except Exception as exc:
            raise ToolExecutionError(
                f"TEOChat request failed: {type(exc).__name__}"
            ) from exc

        return self._tool_success(
            "teochat",
            "paired_optical_analysis",
            result,
        )

    def _execute_m2cd(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        image_t1 = self._file_or_url_arg(arguments, "image_t1_ref")
        image_t2 = self._file_or_url_arg(arguments, "image_t2_ref")

        try:
            result = self._client(
                "m2cd",
                M2CD_SPACE,
            ).predict(
                image_t1=handle_file(image_t1),
                image_t2=handle_file(image_t2),
                api_name=M2CD_API_NAME,
            )
        except Exception as exc:
            raise ToolExecutionError(
                f"M2CD request failed: {type(exc).__name__}"
            ) from exc

        # The exact M2CD output contract is intentionally isolated here.
        # Once its Space is hosted, adapt only this normalization layer.
        try:
            mask = load_numeric_mask(result)
            regions = extract_change_regions(
                mask,
                threshold=M2CD_THRESHOLD,
                max_regions=MAX_CHANGE_REGIONS,
            )
            return self._tool_success(
                "m2cd",
                "sar_change_detection",
                {
                    "regions": regions,
                    "mask_shape": list(mask.shape),
                },
                _raw_mask=mask,
            )
        except Exception:
            # M2CD may return a richer object rather than a raw mask.
            # Preserve the actual result without pretending it is a mask.
            return self._tool_success(
                "m2cd",
                "sar_change_detection",
                result,
            )

    @staticmethod
    def _execute_pseudo_rgb(
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        image_ref = ToolExecutor._file_or_url_arg(
            arguments,
            "image_ref",
        )

        crop = arguments.get("crop")
        if crop is not None and not isinstance(crop, dict):
            raise ToolExecutionError("crop must be an object.")

        local_path = image_ref

        if image_ref.startswith("http://") or image_ref.startswith("https://"):
            from qwen.controller.downloads import download_image_reference
            local_path = download_image_reference(image_ref)

        try:
            output_path = sar_to_pseudo_rgb_file(
                local_path,
                crop=crop,
            )
        except Exception as exc:
            raise ToolExecutionError(
                f"Pseudo-RGB generation failed: {type(exc).__name__}"
            ) from exc

        return ToolExecutor._tool_success(
            "pseudo_rgb",
            "sar_to_pseudo_rgb",
            {
                "image_ref": output_path,
            },
        )

    def execute(
        self,
        tool_name: str,
        arguments: str | dict[str, Any],
    ) -> dict[str, Any]:
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                return {
                    "ok": False,
                    "tool": tool_name,
                    "error": "Invalid tool JSON.",
                }

        if not isinstance(arguments, dict):
            return {
                "ok": False,
                "tool": tool_name,
                "error": "Tool arguments must be an object.",
            }

        dispatch = {
            "geochat": self._execute_geochat,
            "teochat": self._execute_teochat,
            "m2cd": self._execute_m2cd,
            "pseudo_rgb": self._execute_pseudo_rgb,
        }

        handler = dispatch.get(tool_name)
        if handler is None:
            return {
                "ok": False,
                "tool": tool_name,
                "error": f"Unknown tool: {tool_name!r}",
            }

        try:
            result = handler(arguments)

            # Never send a raw numpy mask into Qwen's context.
            if isinstance(result, dict) and "_raw_mask" in result:
                result = dict(result)
                result.pop("_raw_mask", None)

            return result

        except ToolExecutionError as exc:
            logger.error(
                "Tool %s failed: %s",
                tool_name,
                str(exc),
            )
            return {
                "ok": False,
                "tool": tool_name,
                "error": str(exc),
            }
