import logging
from typing import Optional, Any
from fastapi import HTTPException, status

from config import get_settings
from services.qwen import generate_local_satellite_analysis

logger = logging.getLogger("akasha.qwen")


class QwenService:
    """
    Client service for calling the deployed Qwen Hugging Face Gradio Space.
    Architecture:
      FastAPI backend -> Qwen HF Space -> Qwen controller -> GeoChat HF Space -> Qwen synthesizes final response
    """

    def __init__(self, space: Optional[str] = None, token: Optional[str] = None):
        settings = get_settings()
        self.space = space or settings.qwen_space
        self.token = token if token is not None else settings.hf_token
        self._client = None

    def _get_client(self):
        """Lazy initialization of gradio_client.Client."""
        if not self.token:
            logger.error("Attempted to initialize QwenService without HF_TOKEN")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Qwen service is not configured (missing HF_TOKEN)"
            )

        if self._client is None:
            try:
                from gradio_client import Client
                # Initialize gradio client pointing to the Qwen Hugging Face Space
                self._client = Client(
                    self.space,
                    token=self.token,
                )
            except Exception as e:
                # Log safe error without exposing token
                logger.error(f"Failed to connect to Qwen Hugging Face Space: {type(e).__name__}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Failed to connect to upstream Qwen Hugging Face Space"
                )
        return self._client

    def analyze(
        self,
        user_message: str,
        image_url: str,
        max_new_tokens: int = 256,
    ) -> str:
        """
        Call the Qwen Gradio Space endpoint '/ask_akasha' with user prompt and signed image URL.
        Crucial security: image_url MUST be the backend-generated temporary signed URL and NEVER logged.
        """
        if not self.token:
            logger.error("Qwen inference called without HF_TOKEN configured")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Qwen service is not configured (missing HF_TOKEN)"
            )

        client = self._get_client()

        logger.info("Calling Qwen Space endpoint /ask_akasha")

        try:
            controller_message = user_message
            if image_url:
                controller_message = (
                    "A prepared satellite image URL is already supplied as an input. "
                    "Do not ask the user to provide a URL. Call the geochat specialist "
                    "with the supplied image URL, then ground the answer only in its result.\n\n"
                    f"User request: {user_message}"
                )

            result = client.predict(
                user_message=controller_message,
                image_url=image_url,
                max_new_tokens=max_new_tokens,
                api_name="/ask_akasha",
            )
            logger.info("Qwen request completed successfully")

            if result is None:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Upstream Qwen model returned an empty response"
                )

            return str(result).strip()

        except HTTPException:
            raise
        except TimeoutError as te:
            logger.error("Qwen Space call timed out")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Qwen inference timed out"
            )
        # except Exception as e:
        #     err_type = type(e).__name__
        #     err_str = str(e)

        #     if "timeout" in err_str.lower() or "timed out" in err_str.lower():
        #         logger.error("Qwen inference upstream timeout: %s", err_type)
        #         raise HTTPException(
        #             status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        #             detail="Qwen inference timed out"
        #         )

        #     logger.error("Qwen Gradio API error: %s", err_type)

        #     raise HTTPException(
        #         status_code=status.HTTP_502_BAD_GATEWAY,
        #         detail="Qwen vision inference failed"
        #     )
        except Exception as e:
            err_type = type(e).__name__
            err_message = str(e)

            logger.exception("Qwen Gradio inference failed")

            if "ZeroGPU runs limit" in err_message or "ZeroGPU" in err_message:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Your daily AI analysis limit has been reached. Please try again later."
                )

            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="AI analysis is temporarily unavailable. Please try again later."
            )


# Singleton instance
_qwen_service_instance: Optional[QwenService] = None


def get_qwen_service() -> QwenService:
    global _qwen_service_instance
    if _qwen_service_instance is None:
        _qwen_service_instance = QwenService()
    return _qwen_service_instance
