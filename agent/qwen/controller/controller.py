from __future__ import annotations

import json
from typing import Any

from qwen.controller.executor import ToolExecutor
from qwen.controller.model import QwenEngine
from qwen.controller.prompts import SYSTEM_PROMPT
from qwen.controller.tools import get_tool


class QwenController:
    """
    Akasha controller.

    Qwen decides which specialist tool to use.
    ToolExecutor handles communication with the hosted specialist.
    """

    def __init__(
        self,
        qwen: QwenEngine,
        executor: ToolExecutor,
        max_steps: int = 4,
    ) -> None:

        if max_steps < 1:
            raise ValueError(
                "max_steps must be >= 1."
            )

        self.qwen = qwen
        self.executor = executor
        self.max_steps = max_steps

    # ==================================================================
    # Tools
    # ==================================================================

    @staticmethod
    def _get_tools() -> list[dict[str, Any]]:
        return [
            get_tool("geochat"),
        ]

    # ==================================================================
    # Tool result → Qwen message
    # ==================================================================

    @staticmethod
    def _tool_result_message(
        tool_name: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:

        return {
            "role": "tool",
            "content": json.dumps(
                {
                    "tool": tool_name,
                    **result,
                },
                ensure_ascii=False,
            ),
        }

    # ==================================================================
    # Inject backend image URL if Qwen omitted it
    # ==================================================================

    @staticmethod
    def _prepare_tool_arguments(
        tool_name: str,
        arguments: dict[str, Any],
        image_url: str | None,
    ) -> dict[str, Any]:

        prepared = dict(arguments)

        if (
            tool_name == "geochat"
            and "image_url" not in prepared
            and image_url
        ):
            prepared["image_url"] = image_url

        return prepared

    # ==================================================================
    # Main controller loop
    # ==================================================================

    def run(
        self,
        user_message: str,
        image_url: str | None = None,
        max_new_tokens: int = 256,
    ) -> str:

        if not isinstance(
            user_message,
            str,
        ):
            raise TypeError(
                "user_message must be a string."
            )

        user_message = user_message.strip()

        if not user_message:
            raise ValueError(
                "user_message must not be empty."
            )

        if image_url is not None:

            if not isinstance(
                image_url,
                str,
            ):
                raise TypeError(
                    "image_url must be a string."
                )

            image_url = image_url.strip()

            if not image_url:
                image_url = None

        if not 1 <= max_new_tokens <= 1024:
            raise ValueError(
                "max_new_tokens must be between 1 and 1024."
            )

        # ==============================================================
        # Initial conversation
        # ==============================================================

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ]

        tools = self._get_tools()

        # ==============================================================
        # Agent loop
        # ==============================================================

        for step in range(self.max_steps):

            print(
                f"[QwenController] Step "
                f"{step + 1}/{self.max_steps}"
            )

            response = self.qwen.chat_with_tools(
                messages=messages,
                tools=tools,
                max_new_tokens=max_new_tokens,
            )

            print(
                "[QwenController] Qwen response:"
            )
            print(response)

            response_type = response.get(
                "type"
            )

            # ==========================================================
            # Final answer
            # ==========================================================

            if response_type == "text":

                final_text = response.get(
                    "content",
                    "",
                )

                if not isinstance(
                    final_text,
                    str,
                ):
                    raise RuntimeError(
                        "Qwen returned invalid final text."
                    )

                final_text = final_text.strip()

                if not final_text:
                    raise RuntimeError(
                        "Qwen returned an empty final response."
                    )

                return final_text

            # ==========================================================
            # Tool calls
            # ==========================================================

            if response_type != "tool_calls":

                raise RuntimeError(
                    "Unexpected Qwen response type: "
                    f"{response_type!r}"
                )

            tool_calls = response.get(
                "tool_calls",
                [],
            )

            if (
                not isinstance(
                    tool_calls,
                    list,
                )
                or not tool_calls
            ):
                raise RuntimeError(
                    "Qwen returned an empty tool call list."
                )

            # ----------------------------------------------------------
            # Preserve Qwen's tool call in the conversation.
            # ----------------------------------------------------------

            messages.append(
                {
                    "role": "assistant",
                    "content": response.get(
                        "raw",
                        "",
                    ),
                }
            )

            # ==========================================================
            # Execute requested tools
            # ==========================================================

            for call in tool_calls:

                tool_name = call.get(
                    "name"
                )

                if not isinstance(
                    tool_name,
                    str,
                ):
                    continue

                arguments = call.get(
                    "arguments",
                    {},
                )

                if not isinstance(
                    arguments,
                    dict,
                ):
                    arguments = {}

                arguments = self._prepare_tool_arguments(
                    tool_name=tool_name,
                    arguments=arguments,
                    image_url=image_url,
                )

                print(
                    f"[QwenController] Tool: "
                    f"{tool_name}"
                )

                print(
                    f"[QwenController] Arguments: "
                    f"{arguments}"
                )

                result = self.executor.execute(
                    tool_name=tool_name,
                    arguments=arguments,
                )

                print(
                    "[QwenController] Tool result:"
                )
                print(result)

                messages.append(
                    self._tool_result_message(
                        tool_name,
                        result,
                    )
                )

        raise RuntimeError(
            "Qwen controller reached max_steps "
            "without producing a final answer."
        )