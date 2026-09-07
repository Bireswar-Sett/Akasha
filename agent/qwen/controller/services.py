from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


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
    """Authenticated HTTP transport for one Gradio Space."""

    def __init__(self, space: str, api_name: str, token: str | None = None) -> None:
        self.token = token or HF_TOKEN
        if not self.token:
            raise RuntimeError("HF_TOKEN is required for specialist calls")
        self.space = space
        self.api_name = api_name
        self.base_url = self._space_url(space)
        self.timeout = REQUEST_TIMEOUT_SECONDS

    @staticmethod
    def _space_url(space: str) -> str:
        if space.startswith("https://"):
            return space.rstrip("/")
        owner, name = space.split("/", 1)
        return f"https://{owner.lower()}-{name.lower()}.hf.space"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def _upload(self, path: str) -> dict[str, Any]:
        with Path(path).open("rb") as handle:
            response = requests.post(
                f"{self.base_url}/gradio_api/upload",
                headers=self._headers(),
                files={"files": (Path(path).name, handle)},
                timeout=self.timeout,
            )
        response.raise_for_status()
        uploaded = response.json()
        uploaded_path = uploaded[0] if isinstance(uploaded, list) else uploaded
        if isinstance(uploaded_path, str):
            return {"path": uploaded_path, "orig_name": Path(path).name, "meta": {"_type": "gradio.FileData"}}
        if not isinstance(uploaded_path, dict) or not uploaded_path.get("path"):
            raise RuntimeError("Hugging Face upload returned malformed file data")
        uploaded_path.setdefault("meta", {"_type": "gradio.FileData"})
        return uploaded_path

    def _file_data(self, value: str) -> dict[str, Any]:
        parsed = urlparse(value)
        if parsed.scheme in {"http", "https"}:
            suffix = Path(parsed.path).suffix or ".bin"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                temporary_path = handle.name
            try:
                response = requests.get(value, stream=True, timeout=self.timeout)
                response.raise_for_status()
                with Path(temporary_path).open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            output.write(chunk)
                return self._upload(temporary_path)
            finally:
                Path(temporary_path).unlink(missing_ok=True)
        return self._upload(value)

    def predict(self, inputs: dict[str, Any]) -> Any:
        data = {
            key: self._file_data(value)
            if isinstance(value, str) and (value.startswith(("http://", "https://")) or Path(value).is_file())
            else value
            for key, value in inputs.items()
        }
        response = requests.post(
            f"{self.base_url}/gradio_api/call/v2{self.api_name}",
            headers={**self._headers(), "Content-Type": "application/json"},
            json=data,
            timeout=self.timeout,
        )
        response.raise_for_status()
        event_id = response.json().get("event_id")
        if not event_id:
            raise RuntimeError("Hugging Face endpoint returned no event ID")

        with requests.get(
            f"{self.base_url}/gradio_api/call/{self.api_name.lstrip('/')}/{event_id}",
            headers=self._headers(),
            stream=True,
            timeout=self.timeout,
        ) as events:
            events.raise_for_status()
            payload = None
            for line in events.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                value = line[5:].strip()
                if value == "null":
                    continue
                payload = json.loads(value)
                if isinstance(payload, dict) and payload.get("error"):
                    raise RuntimeError("Hugging Face specialist returned an error")
            if payload is None:
                raise RuntimeError("Hugging Face endpoint returned no result")
            return payload


class GeoChatService(HuggingFaceService):
    def analyze(self, image_url: str, prompt: str, max_new_tokens: int = 256) -> Any:
        return self.predict({"image": image_url, "prompt": prompt, "max_new_tokens": max_new_tokens})


class TEOChatService(HuggingFaceService):
    def analyze(self, image_1_url: str, image_2_url: str, prompt: str) -> Any:
        return self.predict({"image_t1": image_1_url, "image_t2": image_2_url, "prompt": prompt})


class M2CDService(HuggingFaceService):
    def detect_change(self, image_t1_url: str, image_t2_url: str) -> Any:
        return self.predict({"image_t1": image_t1_url, "image_t2": image_t2_url})
