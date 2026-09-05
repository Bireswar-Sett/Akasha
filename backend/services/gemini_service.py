import logging
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status
from google import genai

from config import get_settings

logger = logging.getLogger("akasha.gemini")


class GeminiService:
    """
    Text conversation service for AKASHA.

    Gemini handles:
    - normal questions
    - follow-up questions
    - explanations
    - conversation context

    Qwen remains responsible for satellite image analysis.
    """

    def __init__(self):
        settings = get_settings()

        self.api_key = settings.gemini_api_key
        self.model = settings.gemini_model
        self._client = None

    def _get_client(self):
        if not self.api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Gemini service is not configured",
            )

        if self._client is None:
            try:
                self._client = genai.Client(
                    api_key=self.api_key
                )
            except Exception:
                logger.exception("Failed to initialize Gemini client")

                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Failed to initialize Gemini service",
                )

        return self._client

    def chat(
        self,
        message: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:

        client = self._get_client()

        history = history or []

        conversation_parts = []

        for item in history:
            role = item.get("role")

            if role == "user":
                conversation_parts.append(
                    f"User: {item.get('content', '')}"
                )

            elif role == "assistant":
                conversation_parts.append(
                    f"AKASHA: {item.get('content', '')}"
                )

        conversation_context = "\n".join(conversation_parts)

        system_instruction = """
You are AKASHA, a satellite intelligence assistant.

You help users understand:
- satellite imagery
- remote sensing
- NDVI and vegetation indices
- land cover
- vegetation
- earth observation
- geographic and environmental concepts.

Maintain conversational context.

If the user asks a follow-up question, understand references
such as "it", "this", "that", "why is it useful", etc. using
the previous conversation.

Do not claim that you analyzed an image unless an image-analysis
result has actually been provided.

Give clear, concise and technically useful answers.
"""

        if conversation_context:
            prompt = f"""
{system_instruction}

Previous conversation:
{conversation_context}

Current user question:
User: {message}

Answer as AKASHA:
"""
        else:
            prompt = f"""
{system_instruction}

User:
{message}

Answer as AKASHA:
"""

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
            )

            answer = getattr(response, "text", None)

            if not answer:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Gemini returned an empty response",
                )

            return answer.strip()

        except HTTPException:
            raise

        except Exception as exc:
            logger.exception(
                "Gemini request failed: %s",
                type(exc).__name__,
            )

            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="AI chat is temporarily unavailable. Please try again later.",
            )


_gemini_service_instance: Optional[GeminiService] = None


def get_gemini_service() -> GeminiService:
    global _gemini_service_instance

    if _gemini_service_instance is None:
        _gemini_service_instance = GeminiService()

    return _gemini_service_instance