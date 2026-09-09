import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # ──────────────────────────────────────────────────────────────
    # Qwen / Hugging Face
    # ──────────────────────────────────────────────────────────────
    qwen_space: str = os.getenv("QWEN_SPACE", "Bireswar26/Qwen")
    qwen_api_name: str = os.getenv("QWEN_API_NAME", "/analyze")
    hf_token: str = (
        os.getenv("HF_TOKEN", "")
        or os.getenv("HF_INFERENCE_API_KEY", "")
    )

    # ──────────────────────────────────────────────────────────────
    # Signed URL configuration
    # ──────────────────────────────────────────────────────────────
    signed_url_expiration_seconds: int = int(
        os.getenv("SIGNED_URL_EXPIRATION_SECONDS", "1800")
    )

    # ──────────────────────────────────────────────────────────────
    # AWS
    # ──────────────────────────────────────────────────────────────
    aws_region: str = os.getenv("AWS_REGION", "eu-north-1")
    aws_s3_bucket: str = os.getenv(
        "AWS_S3_BUCKET", "akasha-268335032555-eu-north-1-an"
    )
    aws_role_arn: str = os.getenv(
        "AWS_ROLE_ARN",
        "arn:aws:iam::268335032555:role/AkashaBackendRole",
    )
    aws_web_identity_audience: str = os.getenv(
        "AWS_WEB_IDENTITY_AUDIENCE", "https://sts.amazonaws.com"
    )
    aws_session_name: str = os.getenv("AWS_SESSION_NAME", "akasha-backend")

    # ──────────────────────────────────────────────────────────────
    # AWS Cognito
    # ──────────────────────────────────────────────────────────────
    cognito_region: str = os.getenv("COGNITO_REGION", "eu-north-1")
    cognito_user_pool_id: str = os.getenv(
        "COGNITO_USER_POOL_ID", "eu-north-1_R71nMtdeZ"
    )
    cognito_client_id: str = os.getenv(
        "COGNITO_CLIENT_ID", "2dj154bemrmeifpl2n6aob10ck"
    )

    # ──────────────────────────────────────────────────────────────
    # Gemini (text conversations)
    # ──────────────────────────────────────────────────────────────
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    # ──────────────────────────────────────────────────────────────
    # Optional LLM providers (legacy /query fallback)
    # ──────────────────────────────────────────────────────────────
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    qwen_api_key: str = os.getenv("QWEN_API_KEY", "")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")

    # ──────────────────────────────────────────────────────────────
    # Configuration helpers
    # ──────────────────────────────────────────────────────────────
    @property
    def is_qwen_configured(self) -> bool:
        return bool(self.qwen_space and self.hf_token)

    @property
    def is_aws_configured(self) -> bool:
        """
        Return True when the minimum AWS configuration is present.

        On ECS the task role provides credentials automatically via the
        instance metadata service — no explicit key/secret is required.
        We only verify that the target bucket and region are set.
        """
        return bool(self.aws_region and self.aws_s3_bucket)

    @property
    def is_gemini_configured(self) -> bool:
        return bool(self.gemini_api_key)


settings = Settings()


def get_settings() -> Settings:
    return settings
