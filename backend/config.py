import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Ensure .env is loaded
load_dotenv()


@dataclass(frozen=True)
class Settings:
    qwen_space: str = os.getenv("QWEN_SPACE", "AdityaSingh1531/qwen")
    hf_token: str = os.getenv("HF_TOKEN", "") or os.getenv("HF_INFERENCE_API_KEY", "")
    signed_url_expiration_seconds: int = int(os.getenv("SIGNED_URL_EXPIRATION_SECONDS", "1800"))
    firebase_project_id: str = os.getenv("FIREBASE_PROJECT_ID", "akasha-v1")
    firebase_storage_bucket: str = os.getenv("FIREBASE_STORAGE_BUCKET", "akasha-v1.appspot.com")
    firebase_service_account_key_path: str = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY_PATH", "")

    @property
    def is_qwen_configured(self) -> bool:
        return bool(self.qwen_space and self.hf_token)

    @property
    def is_firebase_configured(self) -> bool:
        """Return True only when usable Firebase credentials are available."""

        if (
            self.firebase_service_account_key_path
            and os.path.isfile(self.firebase_service_account_key_path)
        ):
            return True

        google_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if google_credentials and os.path.isfile(google_credentials):
            return True

        return False


settings = Settings()


def get_settings() -> Settings:
    return settings
