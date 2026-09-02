from __future__ import annotations

import json
from typing import Any

from gradio_client import Client, handle_file

from agent.qwen.controller.services import (
    GEOCHAT_SPACE,
    HF_TOKEN,
)


class ToolExecutionError(RuntimeError):
    """Raised when a specialist tool cannot be executed."""


class ToolExecutor:
    """
    Execute Qwen-generated tool calls against hosted
    Hugging Face Gradio Spaces.

    Qwen knows only about tools.
    The executor knows how those tools are hosted.

        Qwen
          ↓
        ToolExecutor
          ↓
        Gradio Client
          ↓
        Hugging Face Space
    """

    def __init__(
        self,
        hf_token: str | None = None,
        geochat_space: str = GEOCHAT_SPACE,
    ) -> None:

        self.hf_token = (
            hf_token
            if hf_token is not None
            else HF_TOKEN
        )

        if not self.hf_token:
            raise ToolExecutionError(
                "HF_TOKEN is required to call "
                "the hosted specialist Spaces."
            )

        self.geochat_space = geochat_space

        print(
            f"[Executor] GeoChat Space: "
            f"{self.geochat_space}"
        )

        self._geochat_client = Client(
            self.geochat_space,
            token=self.hf_token,
        )

    # ==================================================================
    # Argument parsing
    # ==================================================================

    @staticmethod
    def _parse_arguments(
        arguments: str | dict[str, Any],
    ) -> dict[str, Any]:

        if isinstance(arguments, dict):
            return arguments

        if not isinstance(arguments, str):
            raise ToolExecutionError(
                "Tool arguments must be a dictionary "
                "or JSON string."
            )

        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(
                f"Invalid tool JSON: {exc}"
            ) from exc

        if not isinstance(parsed, dict):
            raise ToolExecutionError(
                "Tool arguments must decode to an object."
            )

        return parsed

    # ==================================================================
    # Validation
    # ==================================================================

    @staticmethod
    def _require_url(
        arguments: dict[str, Any],
        name: str,
    ) -> str:

        value = arguments.get(name)

        if not isinstance(value, str):
            raise ToolExecutionError(
                f"{name!r} must be a URL string."
            )

        value = value.strip()

        if not value:
            raise ToolExecutionError(
                f"{name!r} must not be empty."
            )

        return value

    @staticmethod
    def _require_prompt(
        arguments: dict[str, Any],
    ) -> str:

        prompt = arguments.get("prompt")

        if not isinstance(prompt, str):
            raise ToolExecutionError(
                "'prompt' must be a string."
            )

        prompt = prompt.strip()

        if not prompt:
            raise ToolExecutionError(
                "'prompt' must not be empty."
            )

        return prompt

    @staticmethod
    def _max_new_tokens(
        arguments: dict[str, Any],
    ) -> int:

        value = arguments.get(
            "max_new_tokens",
            128,
        )

        if isinstance(value, bool) or not isinstance(value, int):
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
        arguments: dict[str, Any],
    ) -> dict[str, Any]:

        image_url = self._require_url(
            arguments,
            "image_url",
        )

        prompt = self._require_prompt(
            arguments
        )

        max_new_tokens = self._max_new_tokens(
            arguments
        )

        try:

            result = self._geochat_client.predict(
                image=handle_file(image_url),
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                api_name="/geochat",
            )

        except Exception as exc:

            raise ToolExecutionError(
                f"GeoChat request failed: {exc}"
            ) from exc

        return {
            "response": result,
        }

    # ==================================================================
    # Dispatcher
    # ==================================================================

    def execute(
        self,
        tool_name: str,
        arguments: str | dict[str, Any],
    ) -> dict[str, Any]:

        parsed_arguments = self._parse_arguments(
            arguments
        )

        if tool_name == "geochat":

            try:

                result = self._execute_geochat(
                    parsed_arguments
                )

                return {
                    "ok": True,
                    "tool": tool_name,
                    "result": result,
                }

            except ToolExecutionError as exc:

                return {
                    "ok": False,
                    "tool": tool_name,
                    "error": str(exc),
                }

        return {
            "ok": False,
            "tool": tool_name,
            "error": (
                f"Unknown tool: {tool_name!r}"
            ),
        }