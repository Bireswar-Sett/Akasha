from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import HTTPException, status

from config import get_settings

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
        manifest: dict[str, Any] | None = None,
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
        if manifest is None and len(images) > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input Manifest JSON is required when multiple physical files are supplied",
            )

        # The backend supplies the authoritative grouping. References contain
        # stable IDs only; signed URLs remain private to this request.
        if manifest is None:
            manifest = {
                "physical_files": [{"id": "file_0"}],
                "observations": [{"id": "observation_1", "modality": "optical", "image": {"id": "file_0"}}],
                "relationship": {"type": "single"},
            }
        if not isinstance(manifest, dict) or not isinstance(manifest.get("observations"), list):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Input Manifest JSON must contain observations")
        physical_files = manifest.get("physical_files") or [{"id": f"file_{index}"} for index in range(len(images))]
        if len(physical_files) != len(images):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Manifest physical references must match the signed image inputs",
            )
        file_ids = []
        sanitized_files = []
        for index, descriptor in enumerate(physical_files):
            descriptor = dict(descriptor)
            file_id = str(descriptor.get("id") or f"file_{index}")
            if file_id in file_ids:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Manifest physical file IDs must be unique")
            file_ids.append(file_id)
            sanitized_files.append({key: value for key, value in descriptor.items() if key not in {"url", "path"}} | {"id": file_id})

        def sanitize_ref(value: Any) -> dict[str, Any]:
            item = dict(value) if isinstance(value, dict) else {"id": value}
            if "physical_index" in item:
                index = item.pop("physical_index")
                if not isinstance(index, int) or not 0 <= index < len(file_ids):
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Manifest physical reference is invalid")
                item["id"] = file_ids[index]
            ref_id = str(item.get("id") or item.get("file_id") or "")
            # Accept the legacy image_1..image_4 transport aliases but emit
            # only canonical file IDs on the wire.
            if ref_id.startswith("image_") and ref_id[6:].isdigit():
                index = int(ref_id[6:]) - 1
                if 0 <= index < len(file_ids):
                    ref_id = file_ids[index]
                    item["id"] = ref_id
            if ref_id not in file_ids:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Manifest references an unknown physical file")
            return {key: value for key, value in item.items() if key not in {"url", "path", "physical_index", "file_id"}} | {"id": ref_id}

        normalized_observations = []
        for observation in manifest["observations"]:
            item = dict(observation)
            if item.get("modality") == "sar":
                sar = dict(item.get("sar") or {})
                for role in ("vv", "vh"):
                    if role not in sar:
                        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="SAR observations require VV and VH")
                    sar[role] = sanitize_ref(sar[role])
                item["sar"] = sar
                item.pop("image", None)
            else:
                if "image" not in item:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Non-SAR observations require an image")
                item["image"] = sanitize_ref(item["image"])
                item.pop("sar", None)
            normalized_observations.append(item)
        metadata_payload: dict[str, Any] = {
            "physical_files": sanitized_files,
            "observations": normalized_observations,
            "relationship": manifest.get("relationship", relationship or {"type": "single"}),
            "metadata": manifest.get("metadata", {}),
        }
        metadata_payload["capabilities"] = manifest.get("capabilities", {})
        return QwenRequest(
            user_request=user_request,
            images=images,
            metadata=metadata_payload,
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
        manifest: dict[str, Any] | None = None,
    ) -> str:
        urls = list(image_urls or [image_url])
        request = QwenRequestBuilder.build(
            user_request=user_message,
            image_urls=urls,
            image_metadata=image_metadata,
            relationship=relationship,
            manifest=manifest,
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
                # The Space now exposes the normalized-manifest and direct
                # upload inputs as part of its public Gradio contract.
                "manifest_json": json.dumps(request.metadata, ensure_ascii=False),
                "physical_files": [],
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
            # Keep signed URLs and response bodies out of logs, but retain the
            # exception class/message needed to diagnose endpoint mismatches.
            logger.error("Qwen Gradio inference failed: %s: %s", type(exc).__name__, str(exc)[:300])
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
        # Never turn an upstream inference failure into fabricated visual
        # findings. The previous local fallback produced confident-looking
        # reports for zero-byte placeholder images.
        raise HTTPException(
            status_code=status_code or status.HTTP_502_BAD_GATEWAY,
            detail="Qwen inference is temporarily unavailable; inspect backend logs for the upstream error.",
        ) from cause


_qwen_service_instance: Optional[QwenService] = None


def get_qwen_service() -> QwenService:
    global _qwen_service_instance
    if _qwen_service_instance is None:
        _qwen_service_instance = QwenService()
    return _qwen_service_instance
