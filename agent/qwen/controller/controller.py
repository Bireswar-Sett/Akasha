from __future__ import annotations

import json
from typing import Any

from agent.qwen.controller.executor import ToolExecutor
from agent.qwen.controller.model import QwenEngine
from agent.qwen.controller.prompts import SYSTEM_PROMPT
from agent.qwen.controller.tools import get_tool


class QwenController:
    """
    Akasha controller.

    Qwen decides which specialist tool to use.
    ToolExecutor handles communication with the hosted specialist.

        User
          ↓
        Qwen
          ↓
        Tool call
          ↓
        ToolExecutor
          ↓
        Specialist Space
          ↓
        Tool result
          ↓
        Qwen
          ↓
        Final answer
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
    # Tool definitions
    # ==================================================================

    @staticmethod
    def _get_tools() -> list[dict[str, Any]]:
        return [
            get_tool("geochat"),
        ]

    # ==================================================================
    # Tool-result formatting
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
    # Tool-argument injection
    # ==================================================================

    @staticmethod
    def _prepare_tool_arguments(
        tool_name: str,
        arguments: dict[str, Any],
        image_url: str | None,
    ) -> dict[str, Any]:

        prepared = dict(arguments)

        # --------------------------------------------------------------
        # The backend may provide the image separately from the user's
        # natural-language message.
        #
        # This avoids relying on Qwen to reproduce a long signed URL
        # perfectly.
        # --------------------------------------------------------------

        if (
            tool_name == "geochat"
            and "image_url" not in prepared
        ):

            if image_url:
                prepared["image_url"] = image_url

        return prepared

    # ==================================================================
    # Controller
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

        # --------------------------------------------------------------
        # Initial conversation
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Make image availability explicit to Qwen.
        #
        # The actual URL is also injected directly into the tool call
        # if Qwen omits it.
        # --------------------------------------------------------------

        if image_url:

            messages[-1]["content"] = (
                f"{user_message}\n\n"
                "[Available image input]\n"
                f"{image_url}"
            )

        tools = self._get_tools()

        # --------------------------------------------------------------
        # Agent loop
        # --------------------------------------------------------------

        for step in range(
            self.max_steps
        ):

            print(
                f"[QwenController] Step "
                f"{step + 1}/{self.max_steps}"
            )

            response = self.qwen.chat_with_tools(
                messages=messages,
                tools=tools,
            )

            response_type = response.get(
                "type"
            )

            # ----------------------------------------------------------
            # Final text
            # ----------------------------------------------------------

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

                return final_text.strip()

            # ----------------------------------------------------------
            # Tool calls
            # ----------------------------------------------------------

            if response_type != "tool_calls":

                raise RuntimeError(
                    "Unexpected Qwen response type: "
                    f"{response_type!r}"
                )

            tool_calls = response.get(
                "tool_calls",
                [],
            )

            if not isinstance(
                tool_calls,
                list,
            ) or not tool_calls:

                raise RuntimeError(
                    "Qwen returned an empty tool call list."
                )

            # ----------------------------------------------------------
            # Preserve Qwen's tool-call output in conversation.
            # ----------------------------------------------------------

            raw_output = response.get(
                "raw",
                "",
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": raw_output,
                }
            )

            # ----------------------------------------------------------
            # Execute every requested tool.
            # ----------------------------------------------------------

            for call in tool_calls:

                tool_name = call.get(
                    "name"
                )

                arguments = call.get(
                    "arguments",
                    {},
                )

                if not isinstance(
                    tool_name,
                    str,
                ):

                    continue

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