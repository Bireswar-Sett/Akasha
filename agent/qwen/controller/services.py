from __future__ import annotations

import os
from typing import Any

from gradio_client import Client, handle_file


HF_TOKEN = os.getenv("HF_TOKEN")

GEOCHAT_SPACE = os.getenv(
    "GEOCHAT_SPACE",
    "Bireswar26/GeoChat",
)

TEOCHAT_SPACE = os.getenv(
    "TEOCHAT_SPACE",
    "Bireswar26/TEOChat",
)

M2CD_SPACE = os.getenv(
    "M2CD_SPACE",
    "Bireswar26/M2CD",
)

GEOCHAT_API_NAME = os.getenv(
    "GEOCHAT_API_NAME",
    "/geochat",
)

TEOCHAT_API_NAME = os.getenv(
    "TEOCHAT_API_NAME",
    "/teochat",
)

M2CD_API_NAME = os.getenv(
    "M2CD_API_NAME",
    "/m2cd",
)

QWEN_MODEL_ID = os.getenv(
    "QWEN_MODEL_ID",
    "Qwen/Qwen2.5-7B-Instruct",
)

MAX_CONTROLLER_STEPS = int(
    os.getenv("MAX_CONTROLLER_STEPS", "8")
)

REQUEST_TIMEOUT_SECONDS = int(
    os.getenv("REQUEST_TIMEOUT_SECONDS", "180")
)

M2CD_THRESHOLD = float(
    os.getenv("M2CD_THRESHOLD", "0.50")
)

MAX_CHANGE_REGIONS = int(
    os.getenv("MAX_CHANGE_REGIONS", "8")
)


def validate_config() -> None:
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set."
        )


class HuggingFaceService:
    """Authenticated, lazily-created client for one private Space."""

    def __init__(self, space: str, api_name: str) -> None:
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN is required for specialist calls")
        self.space = space
        self.api_name = api_name
        self._client: Client | None = None

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = Client(self.space, token=HF_TOKEN)
        return self._client

    def predict(self, **arguments: Any) -> Any:
        return self.client.predict(api_name=self.api_name, **arguments)


class GeoChatService(HuggingFaceService):
    def analyze(self, image_url: str, prompt: str, max_new_tokens: int = 256) -> Any:
        return self.predict(image=handle_file(image_url), prompt=prompt, max_new_tokens=max_new_tokens)


class TEOChatService(HuggingFaceService):
    def analyze(self, image_1_url: str, image_2_url: str, prompt: str) -> Any:
        return self.predict(image_1=handle_file(image_1_url), image_2=handle_file(image_2_url), prompt=prompt)


class M2CDService(HuggingFaceService):
    def detect_change(self, image_t1_url: str, image_t2_url: str) -> Any:
        return self.predict(image_t1=handle_file(image_t1_url), image_t2=handle_file(image_t2_url))
