from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests


class ToolExecutionError(RuntimeError):
    """Raised when a remote specialist tool cannot be executed."""


class ToolExecutor:
    """
    Execute Qwen-generated tool calls against remote HTTP services.

    Qwen NEVER imports GeoChat, TeoChat, M2CD, etc.

    Instead:

        Qwen
          ↓
        ToolExecutor
          ↓ HTTP
        Specialist Docker service
    """

    def __init__(
        self,
        service_urls: dict[str, str],
        timeout: int = 300,
    ) -> None:

        self.service_urls = dict(
            service_urls
        )

        self.timeout = timeout

    # ==================================================================
    # URL resolution
    # ==================================================================

    def _get_url(
        self,
        tool_name: str,
    ) -> str:

        try:
            return self.service_urls[
                tool_name
            ]
        except KeyError as exc:
            raise ToolExecutionError(
                f"No service URL configured for "
                f"tool {tool_name!r}"
            ) from exc

    # ==================================================================
    # Argument parsing
    # ==================================================================

    @staticmethod
    def _parse_arguments(
        arguments: str | dict[str, Any],
    ) -> dict[str, Any]:

        if isinstance(
            arguments,
            dict,
        ):
            return arguments

        if not isinstance(
            arguments,
            str,
        ):
            raise ToolExecutionError(
                "Tool arguments must be a dictionary "
                "or JSON string."
            )

        try:
            parsed = json.loads(
                arguments
            )
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(
                f"Invalid tool JSON: {exc}"
            ) from exc

        if not isinstance(
            parsed,
            dict,
        ):
            raise ToolExecutionError(
                "Tool arguments must decode to an object."
            )

        return parsed

    # ==================================================================
    # Path validation
    # ==================================================================

    @staticmethod
    def _require_file(
        arguments: dict[str, Any],
        name: str,
    ) -> Path:

        value = arguments.get(
            name
        )

        if not isinstance(
            value,
            str,
        ):
            raise ToolExecutionError(
                f"{name!r} must be a string path."
            )

        path = Path(
            value
        )

        if not path.exists():
            raise ToolExecutionError(
                f"{name!r} does not exist: {path}"
            )

        if not path.is_file():
            raise ToolExecutionError(
                f"{name!r} is not a file: {path}"
            )

        return path

    @staticmethod
    def _require_prompt(
        arguments: dict[str, Any],
    ) -> str:

        prompt = arguments.get(
            "prompt"
        )

        if not isinstance(
            prompt,
            str,
        ):
            raise ToolExecutionError(
                "'prompt' must be a string."
            )

        prompt = prompt.strip()

        if not prompt:
            raise ToolExecutionError(
                "'prompt' must not be empty."
            )

        return prompt

    # ==================================================================
    # Common scalar validation
    # ==================================================================

    @staticmethod
    def _max_new_tokens(
        arguments: dict[str, Any],
    ) -> int:

        value = arguments.get(
            "max_new_tokens",
            128,
        )

        if not isinstance(
            value,
            int,
        ):
            raise ToolExecutionError(
                "max_new_tokens must be an integer."
            )

        if not 1 <= value <= 512:
            raise ToolExecutionError(
                "max_new_tokens must be between 1 and 512."
            )

        return value

    # ==================================================================
    # GeoChat
    # ==================================================================

    def _execute_geochat(
        self,
        url: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:

        image_path = self._require_file(
            arguments,
            "image_path",
        )

        prompt = self._require_prompt(
            arguments
        )

        max_new_tokens = self._max_new_tokens(
            arguments
        )

        with image_path.open(
            "rb"
        ) as image_file:

            files = {
                "image": (
                    image_path.name,
                    image_file,
                    "application/octet-stream",
                )
            }

            data = {
                "prompt": prompt,
                "max_new_tokens": str(
                    max_new_tokens
                ),
            }

            response = requests.post(
                url,
                files=files,
                data=data,
                timeout=self.timeout,
            )

        return self._parse_response(
            response
        )

    # ==================================================================
    # TeoChat
    # ==================================================================

    def _execute_teochat(
        self,
        url: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:

        before_path = self._require_file(
            arguments,
            "before_image",
        )

        after_path = self._require_file(
            arguments,
            "after_image",
        )

        prompt = self._require_prompt(
            arguments
        )

        max_new_tokens = self._max_new_tokens(
            arguments
        )

        with (
            before_path.open("rb") as before_file,
            after_path.open("rb") as after_file,
        ):

            files = {
                "before_image": (
                    before_path.name,
                    before_file,
                    "application/octet-stream",
                ),
                "after_image": (
                    after_path.name,
                    after_file,
                    "application/octet-stream",
                ),
            }

            data = {
                "prompt": prompt,
                "max_new_tokens": str(
                    max_new_tokens
                ),
            }

            response = requests.post(
                url,
                files=files,
                data=data,
                timeout=self.timeout,
            )

        return self._parse_response(
            response
        )

    # ==================================================================
    # M2CD
    # ==================================================================

    def _execute_m2cd(
        self,
        url: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:

        paths = {
            name: self._require_file(
                arguments,
                name,
            )
            for name in (
                "before_vv",
                "before_vh",
                "after_vv",
                "after_vh",
            )
        }

        file_handles = []

        try:

            files = {}

            for field_name, path in paths.items():

                handle = path.open(
                    "rb"
                )

                file_handles.append(
                    handle
                )

                files[field_name] = (
                    path.name,
                    handle,
                    "application/octet-stream",
                )

            response = requests.post(
                url,
                files=files,
                timeout=self.timeout,
            )

        finally:

            for handle in file_handles:
                handle.close()

        return self._parse_response(
            response
        )

    # ==================================================================
    # SAR RGB
    # ==================================================================

    def _execute_sar_rgb(
        self,
        url: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:

        vv_path = self._require_file(
            arguments,
            "vv_path",
        )

        vh_path = self._require_file(
            arguments,
            "vh_path",
        )

        output_path = arguments.get(
            "output_path"
        )

        with (
            vv_path.open("rb") as vv_file,
            vh_path.open("rb") as vh_file,
        ):

            files = {
                "vv": (
                    vv_path.name,
                    vv_file,
                    "application/octet-stream",
                ),
                "vh": (
                    vh_path.name,
                    vh_file,
                    "application/octet-stream",
                ),
            }

            data = {}

            if output_path is not None:

                if not isinstance(
                    output_path,
                    str,
                ):
                    raise ToolExecutionError(
                        "output_path must be a string."
                    )

                data["output_path"] = output_path

            response = requests.post(
                url,
                files=files,
                data=data,
                timeout=self.timeout,
            )

        return self._parse_response(
            response
        )

    # ==================================================================
    # Crop changed region
    # ==================================================================

    def _execute_crop(
        self,
        url: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:

        mask_path = self._require_file(
            arguments,
            "mask_path",
        )

        input_paths = arguments.get(
            "input_paths"
        )

        if not isinstance(
            input_paths,
            list,
        ):
            raise ToolExecutionError(
                "input_paths must be a list."
            )

        if not input_paths:
            raise ToolExecutionError(
                "input_paths must not be empty."
            )

        validated_inputs = [
            self._require_file(
                {
                    "path": path,
                },
                "path",
            )
            for path in input_paths
        ]

        threshold = arguments.get(
            "threshold",
            0.5,
        )

        if not isinstance(
            threshold,
            (int, float),
        ):
            raise ToolExecutionError(
                "threshold must be numeric."
            )

        threshold = float(
            threshold
        )

        if not 0.0 <= threshold <= 1.0:
            raise ToolExecutionError(
                "threshold must be between 0 and 1."
            )

        handles = []

        try:

            files = {
                "mask": (
                    mask_path.name,
                    mask_path.open("rb"),
                    "application/octet-stream",
                )
            }

            handles.append(
                files["mask"][1]
            )

            for index, path in enumerate(
                validated_inputs
            ):

                handle = path.open(
                    "rb"
                )

                handles.append(
                    handle
                )

                files[
                    f"input_{index}"
                ] = (
                    path.name,
                    handle,
                    "application/octet-stream",
                )

            data = {
                "threshold": str(
                    threshold
                )
            }

            response = requests.post(
                url,
                files=files,
                data=data,
                timeout=self.timeout,
            )

        finally:

            for handle in handles:
                handle.close()

        return self._parse_response(
            response
        )

    # ==================================================================
    # Response handling
    # ==================================================================

    @staticmethod
    def _parse_response(
        response: requests.Response,
    ) -> dict[str, Any]:

        if not response.ok:

            try:
                detail = response.json()
            except ValueError:
                detail = response.text

            raise ToolExecutionError(
                f"Remote service returned HTTP "
                f"{response.status_code}: {detail}"
            )

        try:
            result = response.json()
        except ValueError as exc:
            raise ToolExecutionError(
                "Remote service returned non-JSON data."
            ) from exc

        if not isinstance(
            result,
            dict,
        ):
            raise ToolExecutionError(
                "Remote service response must be a JSON object."
            )

        return result

    # ==================================================================
    # Dispatcher
    # ==================================================================

    def execute(
        self,
        tool_name: str,
        arguments: str | dict[str, Any],
    ) -> dict[str, Any]:

        try:

            parsed = self._parse_arguments(
                arguments
            )

            url = self._get_url(
                tool_name
            )

            if tool_name == "geochat":
                result = self._execute_geochat(
                    url,
                    parsed,
                )

            elif tool_name == "teochat":
                result = self._execute_teochat(
                    url,
                    parsed,
                )

            elif tool_name == "m2cd":
                result = self._execute_m2cd(
                    url,
                    parsed,
                )

            elif tool_name == "make_sar_rgb":
                result = self._execute_sar_rgb(
                    url,
                    parsed,
                )

            elif tool_name == "crop_changed_region":
                result = self._execute_crop(
                    url,
                    parsed,
                )

            else:
                return {
                    "ok": False,
                    "tool": tool_name,
                    "error": (
                        f"Unknown tool: {tool_name!r}"
                    ),
                }

            return {
                "ok": True,
                "tool": tool_name,
                "result": result,
            }

        except (
            ToolExecutionError,
            requests.RequestException,
        ) as exc:

            return {
                "ok": False,
                "tool": tool_name,
                "error": str(exc),
            }

    # ==================================================================
    # Health check
    # ==================================================================

    def health_check(
        self,
        tool_name: str,
    ) -> dict[str, Any]:

        try:

            url = self._get_url(
                tool_name
            )

            response = requests.get(
                url,
                timeout=10,
            )

            if not response.ok:
                return {
                    "ok": False,
                    "tool": tool_name,
                    "status_code": response.status_code,
                }

            return {
                "ok": True,
                "tool": tool_name,
                "result": response.json(),
            }

        except Exception as exc:

            return {
                "ok": False,
                "tool": tool_name,
                "error": str(exc),
            }