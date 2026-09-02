from __future__ import annotations

import json
from typing import Any

from models.qwen.controller.executor import ToolExecutor
from models.qwen.controller.model import QwenEngine
from models.qwen.controller.prompts import SYSTEM_PROMPT
from models.qwen.controller.tools import get_tool


class QwenController:
    """
    Akasha controller.

    Qwen decides whether GeoChat should be called.
    Specialist models are accessed only through ToolExecutor.
    """

    def __init__(
        self,
        qwen: QwenEngine,
        executor: ToolExecutor,
    ) -> None:
        self.qwen = qwen
        self.executor = executor

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    @staticmethod
    def _get_tools() -> list[dict[str, Any]]:
        return [
            get_tool("geochat"),
        ]

    # ------------------------------------------------------------------
    # Controller
    # ------------------------------------------------------------------

    def run(
        self,
        user_message: str,
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

        # --------------------------------------------------------------
        # Ask Qwen what to do.
        #
        # NOTE:
        # QwenEngine.chat() currently does not yet expose tools.
        # We therefore stop here until we update model.py to support
        # structured tool calling.
        # --------------------------------------------------------------

        raise NotImplementedError(
            "QwenEngine.chat() needs tool-calling support "
            "before QwenController can run."
        )