from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from gradio_client import Client, handle_file

from qwen.controller.downloads import download_image_reference
from qwen.controller.image_processing import (
    combine_sar_channels_file,
    extract_change_regions,
    load_numeric_mask,
    sar_to_pseudo_rgb_file,
)
from qwen.controller.schemas import (
    AnalysisRequest,
    BoundingBox,
    InputManifest,
    Observation,
    ToolName,
)
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

    def __init__(
        self,
        message: str,
        status: str = "upstream_error",
    ) -> None:
        super().__init__(message)
        self.status = status


class ToolExecutor:
    """
    The only layer that knows how specialist tools are physically invoked.

    Qwen / planner:
        WHAT should happen

    ToolExecutor:
        HOW that operation is physically executed

    The executor also owns the per-request artifact registry so that
    downstream tool calls can consume outputs from earlier steps.
    """

    def __init__(
        self,
        hf_token: str | None = None,
    ) -> None:
        self.hf_token = (
            hf_token
            if hf_token is not None
            else HF_TOKEN
        )

        self._clients: dict[str, Client] = {}

        # Bound for one controller request.
        self._request: AnalysisRequest | None = None

        # Generated intermediate artifacts.
        # Example:
        #   "artifact:pseudo_rgb" -> "/tmp/..."
        self._artifacts: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Request / artifact context
    # ------------------------------------------------------------------

    def bind_request(
        self,
        request: AnalysisRequest,
    ) -> None:
        """
        Bind the logical manifest for the current controller request.

        This does not mutate the user's request.
        It simply gives the executor enough context to resolve logical
        observation IDs emitted by the planner.
        """
        self._request = request
        self._artifacts.clear()

    def clear_context(self) -> None:
        """Release request-local references and generated artifact state."""
        self._request = None
        self._artifacts.clear()

    def register_artifact(
        self,
        name: str,
        value: Any,
    ) -> None:
        if not name or not name.strip():
            raise ValueError("artifact name must not be empty")

        self._artifacts[name.strip()] = value

    def _manifest(self) -> InputManifest:
        if self._request is None:
            raise ToolExecutionError(
                "No analysis request is bound to the executor.",
                "invalid_input",
            )

        manifest = self._request.manifest

        if manifest is None:
            raise ToolExecutionError(
                "Analysis request has no input manifest.",
                "invalid_input",
            )

        if isinstance(manifest, dict):
            try:
                manifest = InputManifest.model_validate(manifest)
            except ValueError as exc:
                raise ToolExecutionError(
                    f"Invalid input manifest: {exc}",
                    "invalid_input",
                ) from exc

        return manifest

    # ------------------------------------------------------------------
    # Errors / retries
    # ------------------------------------------------------------------

    @staticmethod
    def _failure_status(exc: Exception) -> str:
        name = type(exc).__name__.lower()
        message = str(exc).lower()

        if "timeout" in name or "timeout" in message:
            return "timeout"

        if any(
            value in message
            for value in (
                "401",
                "403",
                "unauthorized",
                "forbidden",
                "authentication",
            )
        ):
            return "authentication_error"

        if any(
            value in message
            for value in (
                "400",
                "invalid input",
                "validation",
                "unsupported endpoint",
            )
        ):
            return "invalid_input"

        if any(
            value in message
            for value in (
                "503",
                "502",
                "500",
                "space is unavailable",
                "sleeping",
                "startup",
                "connection",
                "temporarily",
            )
        ):
            return "unavailable"

        return "upstream_error"

    @staticmethod
    def _safe_error(
        tool_name: str,
        status: str,
    ) -> str:
        return (
            f"{tool_name} specialist is currently "
            f"{status.replace('_', ' ')}."
        )

    def _remote_predict(
        self,
        key: str,
        space: str,
        tool_name: str,
        **kwargs: Any,
    ) -> Any:
        if not self.hf_token:
            raise ToolExecutionError(
                f"{tool_name} specialist authentication is not configured.",
                "authentication_error",
            )

        attempts = 2

        for attempt in range(attempts):
            try:
                return self._client(
                    key,
                    space,
                ).predict(**kwargs)

            except Exception as exc:
                failure_status = self._failure_status(exc)

                transient = failure_status in {
                    "timeout",
                    "unavailable",
                    "upstream_error",
                }

                if (
                    transient
                    and attempt + 1 < attempts
                ):
                    time.sleep(
                        0.5 * (2 ** attempt)
                    )
                    continue

                raise ToolExecutionError(
                    self._safe_error(
                        tool_name,
                        failure_status,
                    ),
                    failure_status,
                ) from exc

        raise ToolExecutionError(
            f"{tool_name} specialist execution failed.",
            "upstream_error",
        )

    def _client(
        self,
        key: str,
        space: str,
    ) -> Client:
        if key not in self._clients:
            logger.info(
                "Initializing HF Space client: %s",
                space,
            )

            self._clients[key] = Client(
                space,
                token=self.hf_token,
            )

        return self._clients[key]

    # ------------------------------------------------------------------
    # Generic argument validation
    # ------------------------------------------------------------------

    @staticmethod
    def _string_arg(
        arguments: dict[str, Any],
        key: str,
    ) -> str:
        value = arguments.get(key)

        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise ToolExecutionError(
                f"Tool argument {key!r} must be a non-empty string.",
                "invalid_input",
            )

        return value.strip()

    @staticmethod
    def _token_limit(
        arguments: dict[str, Any],
        default: int,
        maximum: int,
    ) -> int:
        value = arguments.get(
            "max_new_tokens",
            default,
        )

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise ToolExecutionError(
                "max_new_tokens must be an integer.",
                "invalid_input",
            )

        if not 1 <= value <= maximum:
            raise ToolExecutionError(
                f"max_new_tokens must be between 1 and {maximum}.",
                "invalid_input",
            )

        return value

    @staticmethod
    def _prompt(
        arguments: dict[str, Any],
    ) -> str:
        return ToolExecutor._string_arg(
            arguments,
            "prompt",
        )

    @staticmethod
    def _file_or_url_arg(
        arguments: dict[str, Any],
        key: str,
    ) -> str:
        value = ToolExecutor._string_arg(
            arguments,
            key,
        )

        return ToolExecutor._validate_file_or_url(
            value,
            key,
        )

    @staticmethod
    def _validate_file_or_url(
        value: str,
        key: str,
    ) -> str:
        if value.startswith("https://"):
            return value

        if value.startswith("http://"):
            return value

        try:
            if Path(value).is_file():
                return value
        except OSError:
            pass

        raise ToolExecutionError(
            f"{key!r} must be an HTTPS URL or an existing local file.",
            "invalid_input",
        )

    # ------------------------------------------------------------------
    # Manifest resolution
    # ------------------------------------------------------------------

    def _observation(
        self,
        observation_id: str,
    ) -> Observation:
        observation_id = self._validate_id(
            observation_id,
            "observation_id",
        )

        manifest = self._manifest()

        for observation in manifest.observations:
            if observation.id == observation_id:
                return observation

        raise ToolExecutionError(
            f"Unknown observation ID: {observation_id!r}.",
            "invalid_input",
        )

    @staticmethod
    def _validate_id(
        value: Any,
        field_name: str,
    ) -> str:
        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise ToolExecutionError(
                f"{field_name} must be a non-empty string.",
                "invalid_input",
            )

        return value.strip()

    def _physical_reference(
        self,
        image_id: str,
    ) -> str:
        image_id = self._validate_id(
            image_id,
            "image_id",
        )

        manifest = self._manifest()

        for image in manifest.physical_files:
            if image.image_id == image_id:
                return image.url

        raise ToolExecutionError(
            f"Unknown physical image ID: {image_id!r}.",
            "invalid_input",
        )

    def _observation_image(
        self,
        observation_id: str,
    ) -> str:
        observation = self._observation(
            observation_id,
        )

        if observation.image is None:
            raise ToolExecutionError(
                f"Observation {observation_id!r} does not contain "
                "a single-image reference.",
                "invalid_input",
            )

        return observation.image.url

    def _observation_sar_channels(
        self,
        observation_id: str,
    ) -> tuple[str, str]:
        observation = self._observation(
            observation_id,
        )

        if observation.sar is None:
            raise ToolExecutionError(
                f"Observation {observation_id!r} is not a SAR observation.",
                "invalid_input",
            )

        return (
            observation.sar.vv.url,
            observation.sar.vh.url,
        )

    def _resolve_image_reference(
        self,
        arguments: dict[str, Any],
        *,
        image_ref_key: str = "image_ref",
        observation_key: str = "observation_id",
    ) -> str:
        """
        Resolve the planner's logical references into an actual authorized
        file/URL.

        Supported planner forms:

            image_ref="artifact:pseudo_rgb"

        or:

            observation_id="observation_1"

        Legacy direct references remain supported.
        """

        artifact_ref = arguments.get(
            image_ref_key
        )

        if isinstance(
            artifact_ref,
            str,
        ) and artifact_ref.startswith("artifact:"):
            artifact = self._artifacts.get(
                artifact_ref
            )

            if not isinstance(
                artifact,
                str,
            ):
                raise ToolExecutionError(
                    f"Unknown generated artifact: {artifact_ref!r}.",
                    "invalid_input",
                )

            return self._validate_file_or_url(
                artifact,
                image_ref_key,
            )

        if isinstance(
            artifact_ref,
            str,
        ) and artifact_ref.strip():
            return self._validate_file_or_url(
                artifact_ref.strip(),
                image_ref_key,
            )

        observation_id = arguments.get(
            observation_key
        )

        if observation_id:
            return self._observation_image(
                observation_id
            )

        raise ToolExecutionError(
            f"Tool requires {image_ref_key!r} or "
            f"{observation_key!r}.",
            "invalid_input",
        )

    # ------------------------------------------------------------------
    # Tool result helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # GeoChat
    # ------------------------------------------------------------------

    def _execute_geochat(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        image_ref = self._resolve_image_reference(
            arguments,
        )

        prompt = self._prompt(
            arguments
        )

        bounding_box = arguments.get(
            "bounding_box"
        )

        if bounding_box is not None:
            try:
                bounding_box = BoundingBox.model_validate(
                    bounding_box
                )
            except ValueError as exc:
                raise ToolExecutionError(
                    f"Invalid bounding_box: {exc}",
                    "invalid_input",
                ) from exc

        max_tokens = self._token_limit(
            arguments,
            256,
            512,
        )

        result = self._remote_predict(
            "geochat",
            GEOCHAT_SPACE,
            "GeoChat",
            image=handle_file(image_ref),
            prompt=prompt,
            max_new_tokens=max_tokens,
            api_name=GEOCHAT_API_NAME,
        )

        operation = str(
            arguments.get(
                "operation",
                "region_grounded_analysis"
                if bounding_box is not None
                else "single_image_analysis",
            )
        )

        return self._tool_success(
            "geochat",
            operation,
            result,
            bounding_box=(
                bounding_box.model_dump()
                if bounding_box is not None
                else None
            ),
        )

    # ------------------------------------------------------------------
    # TEOChat
    # ------------------------------------------------------------------

    def _execute_teochat(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        image_1 = None
        image_2 = None

        # Canonical planner representation.
        observation_1_id = arguments.get(
            "observation_1_id"
        )
        observation_2_id = arguments.get(
            "observation_2_id"
        )

        if observation_1_id and observation_2_id:
            observation_1 = self._observation(
                observation_1_id
            )
            observation_2 = self._observation(
                observation_2_id
            )

            if observation_1.image is None:
                raise ToolExecutionError(
                    (
                        f"TEOChat observation "
                        f"{observation_1_id!r} does not contain "
                        "a single optical/multispectral image."
                    ),
                    "invalid_input",
                )

            if observation_2.image is None:
                raise ToolExecutionError(
                    (
                        f"TEOChat observation "
                        f"{observation_2_id!r} does not contain "
                        "a single optical/multispectral image."
                    ),
                    "invalid_input",
                )

            image_1 = observation_1.image.url
            image_2 = observation_2.image.url

        else:
            # Legacy direct references.
            image_1 = self._file_or_url_arg(
                arguments,
                "image_1_ref",
            )
            image_2 = self._file_or_url_arg(
                arguments,
                "image_2_ref",
            )

        prompt = self._prompt(
            arguments
        )

        max_tokens = self._token_limit(
            arguments,
            384,
            768,
        )

        result = self._remote_predict(
            "teochat",
            TEOCHAT_SPACE,
            "TEOChat",
            image_t1=handle_file(image_1),
            image_t2=handle_file(image_2),
            prompt=prompt,
            max_new_tokens=max_tokens,
            api_name=TEOCHAT_API_NAME,
        )

        operation = str(
            arguments.get(
                "operation",
                "temporal_analysis",
            )
        )

        return self._tool_success(
            "teochat",
            operation,
            result,
        )

    # ------------------------------------------------------------------
    # M²CD
    # ------------------------------------------------------------------

    def _localize(
        self,
        reference: str,
    ) -> str:
        """
        Make a remote reference locally available when deterministic
        processing requires filesystem access.
        """
        if reference.startswith("https://"):
            try:
                return download_image_reference(
                    reference
                )
            except Exception as exc:
                raise ToolExecutionError(
                    "Authorized image download failed.",
                    "upstream_error",
                ) from exc

        if reference.startswith("http://"):
            try:
                return download_image_reference(
                    reference
                )
            except Exception as exc:
                raise ToolExecutionError(
                    "Authorized image download failed.",
                    "upstream_error",
                ) from exc

        return self._validate_file_or_url(
            reference,
            "image_ref",
        )

    def _execute_m2cd(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute M²CD.

        Canonical planner input:

            observation_1_id
            observation_2_id

        Legacy four-reference input is also retained.
        """

        observation_1_id = arguments.get(
            "observation_1_id"
        )
        observation_2_id = arguments.get(
            "observation_2_id"
        )

        if observation_1_id and observation_2_id:
            t1_vv, t1_vh = self._observation_sar_channels(
                observation_1_id
            )
            t2_vv, t2_vh = self._observation_sar_channels(
                observation_2_id
            )

            t1_vv = self._localize(t1_vv)
            t1_vh = self._localize(t1_vh)
            t2_vv = self._localize(t2_vv)
            t2_vh = self._localize(t2_vh)

            # Keep channel combination inside the executor. M²CD should
            # never receive Qwen-invented filesystem paths.
            image_t1 = combine_sar_channels_file(
                t1_vv,
                t1_vh,
            )

            image_t2 = combine_sar_channels_file(
                t2_vv,
                t2_vh,
            )

        elif all(
            key in arguments
            for key in (
                "image_t1_vv_ref",
                "image_t1_vh_ref",
                "image_t2_vv_ref",
                "image_t2_vh_ref",
            )
        ):
            image_t1 = combine_sar_channels_file(
                self._localize(
                    self._file_or_url_arg(
                        arguments,
                        "image_t1_vv_ref",
                    )
                ),
                self._localize(
                    self._file_or_url_arg(
                        arguments,
                        "image_t1_vh_ref",
                    )
                ),
            )

            image_t2 = combine_sar_channels_file(
                self._localize(
                    self._file_or_url_arg(
                        arguments,
                        "image_t2_vv_ref",
                    )
                ),
                self._localize(
                    self._file_or_url_arg(
                        arguments,
                        "image_t2_vh_ref",
                    )
                ),
            )

        else:
            image_t1 = self._file_or_url_arg(
                arguments,
                "image_t1_ref",
            )
            image_t2 = self._file_or_url_arg(
                arguments,
                "image_t2_ref",
            )

        result = self._remote_predict(
            "m2cd",
            M2CD_SPACE,
            "M2CD",
            image_t1=handle_file(image_t1),
            image_t2=handle_file(image_t2),
            api_name=M2CD_API_NAME,
        )

        # Keep the exact external M²CD output isolated here.
        #
        # If it is a numeric mask, normalize it into spatial evidence.
        # Otherwise preserve the actual returned object rather than
        # pretending that an arbitrary response is a mask.
        try:
            mask = load_numeric_mask(
                result
            )

            regions = extract_change_regions(
                mask,
                threshold=M2CD_THRESHOLD,
                max_regions=MAX_CHANGE_REGIONS,
            )

            normalized = {
                "regions": regions,
                "mask_shape": list(
                    mask.shape
                ),
            }

            # Never pass numpy arrays into the LLM context.
            return self._tool_success(
                "m2cd",
                str(
                    arguments.get(
                        "operation",
                        "change_detection",
                    )
                ),
                normalized,
            )

        except Exception:
            return self._tool_success(
                "m2cd",
                str(
                    arguments.get(
                        "operation",
                        "change_detection",
                    )
                ),
                result,
            )

    # ------------------------------------------------------------------
    # Pseudo-RGB
    # ------------------------------------------------------------------

    def _execute_pseudo_rgb(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Generate GeoChat-compatible SAR pseudo-RGB.

        Canonical planner input:

            observation_id

        The generated result is registered automatically as:

            artifact:pseudo_rgb

        Legacy direct references remain supported.
        """

        observation_id = arguments.get(
            "observation_id"
        )

        if observation_id:
            vv_ref, vh_ref = self._observation_sar_channels(
                observation_id
            )

            vv_ref = self._localize(
                vv_ref
            )
            vh_ref = self._localize(
                vh_ref
            )

            source_path = combine_sar_channels_file(
                vv_ref,
                vh_ref,
            )

        else:
            image_ref = arguments.get(
                "image_ref"
            )

            vv_ref = arguments.get(
                "vv_ref"
            )
            vh_ref = arguments.get(
                "vh_ref"
            )

            if (
                isinstance(image_ref, str)
                and image_ref.strip()
            ):
                image_ref = self._resolve_image_reference(
                    arguments,
                )

                source_path = self._localize(
                    image_ref
                )

            elif (
                isinstance(vv_ref, str)
                and isinstance(vh_ref, str)
            ):
                vv_ref = self._file_or_url_arg(
                    arguments,
                    "vv_ref",
                )

                vh_ref = self._file_or_url_arg(
                    arguments,
                    "vh_ref",
                )

                vv_ref = self._localize(
                    vv_ref
                )

                vh_ref = self._localize(
                    vh_ref
                )

                source_path = combine_sar_channels_file(
                    vv_ref,
                    vh_ref,
                )

            else:
                raise ToolExecutionError(
                    "Pseudo-RGB requires an observation_id, image_ref, "
                    "or VV/VH references.",
                    "invalid_input",
                )

        crop = arguments.get(
            "crop"
        )

        if crop is not None and not isinstance(
            crop,
            dict,
        ):
            raise ToolExecutionError(
                "crop must be an object.",
                "invalid_input",
            )

        try:
            output_path = sar_to_pseudo_rgb_file(
                source_path,
                crop=crop,
            )

        except Exception as exc:
            raise ToolExecutionError(
                (
                    "Pseudo-RGB generation failed: "
                    f"{type(exc).__name__}"
                ),
                "upstream_error",
            ) from exc

        # Register the canonical artifact name expected by the planner.
        self.register_artifact(
            "artifact:pseudo_rgb",
            output_path,
        )

        return self._tool_success(
            "pseudo_rgb",
            str(
                arguments.get(
                    "operation",
                    "sar_to_pseudo_rgb",
                )
            ),
            {
                "image_ref": output_path,
                "artifact_ref": "artifact:pseudo_rgb",
            },
        )

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def execute(
        self,
        tool_name: str | ToolName,
        arguments: str | dict[str, Any],
        request: AnalysisRequest | None = None,
    ) -> dict[str, Any]:
        """
        Execute exactly one validated tool operation.

        `request` is optional for backward compatibility. Prefer binding
        a request once with bind_request(), or pass it here.

        The executor never executes unknown tools.
        """

        if request is not None:
            self.bind_request(
                request
            )

        if isinstance(
            tool_name,
            ToolName,
        ):
            tool_name = tool_name.value

        if not isinstance(
            tool_name,
            str,
        ):
            return {
                "ok": False,
                "tool": str(tool_name),
                "error": "Tool name must be a string.",
                "status": "invalid_input",
            }

        tool_name = tool_name.strip()

        if isinstance(
            arguments,
            str,
        ):
            try:
                arguments = json.loads(
                    arguments
                )
            except json.JSONDecodeError:
                return {
                    "ok": False,
                    "tool": tool_name,
                    "error": "Invalid tool JSON.",
                    "status": "invalid_input",
                }

        if not isinstance(
            arguments,
            dict,
        ):
            return {
                "ok": False,
                "tool": tool_name,
                "error": "Tool arguments must be an object.",
                "status": "invalid_input",
            }

        dispatch = {
            "geochat": self._execute_geochat,
            "teochat": self._execute_teochat,
            "m2cd": self._execute_m2cd,
            "pseudo_rgb": self._execute_pseudo_rgb,
        }

        handler = dispatch.get(
            tool_name
        )

        if handler is None:
            return {
                "ok": False,
                "tool": tool_name,
                "error": (
                    f"Unknown tool: {tool_name!r}"
                ),
                "status": "invalid_input",
            }

        try:
            result = handler(
                arguments
            )

            # Defensive cleanup: raw numpy masks must never reach Qwen.
            if (
                isinstance(result, dict)
                and "_raw_mask" in result
            ):
                result = dict(result)
                result.pop(
                    "_raw_mask",
                    None,
                )

            return result

        except ToolExecutionError as exc:
            logger.warning(
                "Tool %s failed with status %s",
                tool_name,
                exc.status,
            )

            return {
                "ok": False,
                "tool": tool_name,
                "error": str(exc),
                "status": exc.status,
            }

        except Exception as exc:
            failure_status = self._failure_status(
                exc
            )

            logger.warning(
                "Tool %s failed with status %s",
                tool_name,
                failure_status,
            )

            return {
                "ok": False,
                "tool": tool_name,
                "error": self._safe_error(
                    tool_name,
                    failure_status,
                ),
                "status": failure_status,
            }