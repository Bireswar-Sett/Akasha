from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import HTTPException, status

from config import get_settings
from services.qwen import generate_local_satellite_analysis

logger = logging.getLogger("akasha.qwen")


@dataclass(frozen=True)
class QwenImageInput:
    image_id: str
    signed_url: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QwenRequest:
    user_request: str
    images: tuple[QwenImageInput, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


class QwenRequestBuilder:
    """Build the private backend-to-Qwen request without exposing URLs in logs."""

    @staticmethod
    def build(
        user_request: str,
        image_urls: list[str],
        image_metadata: list[dict[str, Any]] | None = None,
        relationship: dict[str, Any] | None = None,
    ) -> QwenRequest:
        if not 1 <= len(image_urls) <= 4:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Qwen supports between one and four image inputs",
            )
        if any(not isinstance(url, str) or not url.strip() for url in image_urls):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Qwen image inputs must be non-empty signed URLs",
            )
        metadata = image_metadata or [{} for _ in image_urls]
        if len(metadata) != len(image_urls):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image metadata must match the number of image inputs",
            )
        images = tuple(
            QwenImageInput(
                image_id=f"image_{index}",
                signed_url=url,
                metadata=dict(item),
            )
            for index, (url, item) in enumerate(zip(image_urls, metadata), start=1)
        )
        return QwenRequest(
            user_request=user_request,
            images=images,
            metadata={"images": [image.metadata for image in images], "pair_metadata": relationship},
        )


class QwenService:
    """Authenticated client for the signed-URL-only Qwen Gradio Space."""

    def __init__(self, space: Optional[str] = None, token: Optional[str] = None):
        settings = get_settings()
        self.space = space or settings.qwen_space
        self.api_name = settings.qwen_api_name
        self.token = token if token is not None else settings.hf_token
        self._client = None

    def _get_client(self):
        if not self.token:
            logger.error("Attempted to initialize QwenService without HF_TOKEN")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Qwen service is not configured (missing HF_TOKEN)",
            )
        if self._client is None:
            try:
                from gradio_client import Client
                self._client = Client(self.space, token=self.token)
            except Exception as exc:
                logger.error("Failed to connect to Qwen Space: %s", type(exc).__name__)
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Failed to connect to upstream Qwen Hugging Face Space",
                ) from exc
        return self._client

    def analyze(
        self,
        user_message: str,
        image_url: str,
        max_new_tokens: int = 256,
        *,
        image_urls: list[str] | None = None,
        image_metadata: list[dict[str, Any]] | None = None,
        relationship: dict[str, Any] | None = None,
    ) -> str:
        urls = list(image_urls or [image_url])
        request = QwenRequestBuilder.build(
            user_request=user_message,
            image_urls=urls,
            image_metadata=image_metadata,
            relationship=relationship,
        )
        client = self._get_client()
        logger.info("Calling Qwen Space endpoint %s for %d image input(s)", self.api_name, len(urls))

        try:
            arguments: dict[str, Any] = {
                "user_request": request.user_request,
                "url_1": request.images[0].signed_url,
                "url_2": request.images[1].signed_url if len(request.images) > 1 else "",
                "url_3": request.images[2].signed_url if len(request.images) > 2 else "",
                "url_4": request.images[3].signed_url if len(request.images) > 3 else "",
                "api_name": self.api_name,
            }
            result = client.predict(**arguments)
            if result is None or not str(result).strip():
                raise RuntimeError("empty response")
            logger.info("Qwen request completed successfully")
            return self._response_text(result)
        except HTTPException:
            raise
        except TimeoutError as exc:
            logger.error("Qwen Space call timed out")
            return self._local_fallback(request, max_new_tokens, status_code=status.HTTP_504_GATEWAY_TIMEOUT, cause=exc)
        except Exception as exc:
            message = str(exc)
            logger.error("Qwen Gradio inference failed: %s", type(exc).__name__)
            if "ZeroGPU runs limit" in message or "ZeroGPU" in message:
                return self._local_fallback(request, max_new_tokens, status_code=status.HTTP_429_TOO_MANY_REQUESTS, cause=exc)
            return self._local_fallback(request, max_new_tokens, cause=exc)

    @staticmethod
    def _response_text(result: Any) -> str:
        text = str(result).strip()
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return text
        if isinstance(payload, dict) and isinstance(payload.get("answer"), str):
            return payload["answer"].strip()
        return text

    @staticmethod
    def _local_fallback(request: QwenRequest, max_new_tokens: int, status_code: int | None = None, cause: Exception | None = None) -> str:
        images = [{"filename": image.image_id, "bytes": b""} for image in request.images]
        fallback = generate_local_satellite_analysis(request.user_request, images)
        if fallback and fallback.get("response"):
            return str(fallback["response"]).strip()
        if status_code is not None:
            raise HTTPException(status_code=status_code, detail="Qwen inference is temporarily unavailable") from cause
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="AI analysis is temporarily unavailable") from cause


_qwen_service_instance: Optional[QwenService] = None


def get_qwen_service() -> QwenService:
    global _qwen_service_instance
    if _qwen_service_instance is None:
        _qwen_service_instance = QwenService()
    return _qwen_service_instance
