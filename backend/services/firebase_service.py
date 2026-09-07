from __future__ import annotations

import datetime
import logging
import os
from typing import Any, Dict, Optional

import firebase_admin
from fastapi import HTTPException, status
from firebase_admin import auth, credentials, storage

from config import get_settings
from services.image_metadata import normalize_storage_metadata


logger = logging.getLogger("akasha.firebase")


class FirebaseStorageService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._init_firebase()

    # ------------------------------------------------------------------
    # Firebase initialization
    # ------------------------------------------------------------------

    def _init_firebase(self) -> None:
        """Idempotently initialize Firebase Admin SDK."""
        if firebase_admin._apps:
            return

        key_path = self.settings.firebase_service_account_key_path
        bucket_name = self.settings.firebase_storage_bucket
        project_id = self.settings.firebase_project_id

        options: dict[str, Any] = {}

        if bucket_name:
            options["storageBucket"] = bucket_name

        if project_id:
            options["projectId"] = project_id

        try:
            if key_path and os.path.isfile(key_path):
                cred = credentials.Certificate(
                    key_path
                )

                firebase_admin.initialize_app(
                    cred,
                    options,
                )

                logger.info(
                    "Firebase Admin initialized with service account credentials"
                )

            elif os.getenv(
                "GOOGLE_APPLICATION_CREDENTIALS"
            ):
                cred = credentials.ApplicationDefault()

                firebase_admin.initialize_app(
                    cred,
                    options,
                )

                logger.info(
                    "Firebase Admin initialized with Application Default Credentials"
                )

            else:
                firebase_admin.initialize_app(
                    options=options
                )

                logger.info(
                    "Firebase Admin initialized with default credentials"
                )

        except Exception:
            logger.exception(
                "Firebase Admin initialization failed"
            )
            raise

    # ------------------------------------------------------------------
    # Storage helpers
    # ------------------------------------------------------------------

    def get_bucket(self):
        """Retrieve the configured Firebase Storage bucket."""
        try:
            return storage.bucket(
                self.settings.firebase_storage_bucket
            )

        except Exception as exc:
            logger.error(
                "Failed to acquire Firebase Storage bucket: %s",
                type(exc).__name__,
            )

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Storage service unavailable",
            ) from exc

    def _get_blob(
        self,
        image_path: str,
    ):
        """
        Retrieve and load the actual Storage object.

        IMPORTANT:
        bucket.blob(path) creates a local Blob reference but does not
        guarantee that object metadata has been loaded.

        bucket.get_blob(path) retrieves the actual object resource,
        including custom metadata.
        """
        clean_path = self.validate_image_path(
            image_path
        )

        try:
            blob = self.get_bucket().get_blob(
                clean_path
            )

        except Exception as exc:
            logger.error(
                "Failed to retrieve Storage object: %s",
                type(exc).__name__,
            )

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to access storage object",
            ) from exc

        if blob is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="The requested image object was not found in storage",
            )

        return blob

    # ------------------------------------------------------------------
    # Path validation
    # ------------------------------------------------------------------

    def validate_image_path(
        self,
        image_path: str,
    ) -> str:
        """
        Validate Storage object path and guard against traversal.

        Allowed namespaces:

            users/<uid>/...
            satellite_images/<uid>/...
        """

        if (
            not image_path
            or not isinstance(
                image_path,
                str,
            )
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image path must be a non-empty string",
            )

        clean_path = (
            image_path
            .strip()
            .replace("\\", "/")
        )

        if (
            ".." in clean_path
            or clean_path.startswith("/")
            or "\0" in clean_path
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Invalid image path: directory traversal "
                    "or illegal characters detected"
                ),
            )

        valid_extensions = (
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff",
            ".webp",
        )

        if not any(
            clean_path.lower().endswith(ext)
            for ext in valid_extensions
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Invalid image path: unsupported image file format"
                ),
            )

        return clean_path

    # ------------------------------------------------------------------
    # Authorization
    # ------------------------------------------------------------------

    def verify_user_authorization(
        self,
        user_id: str,
        image_path: str,
    ) -> None:
        """
        Ensure the authenticated user owns the requested Storage path.
        """

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User identity could not be verified",
            )

        clean_path = self.validate_image_path(
            image_path
        )

        parts = clean_path.strip("/").split("/")

        if len(parts) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Invalid image path format. Expected "
                    "'<category>/<user_id>/<filename>'"
                ),
            )

        prefix = parts[0]
        path_user_id = parts[1]

        allowed_prefixes = {
            "users",
            "satellite_images",
        }

        if prefix not in allowed_prefixes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: unauthorized storage namespace",
            )

        if path_user_id != user_id:
            logger.warning(
                "Storage authorization failure for authenticated user"
            )

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Access denied: you do not have permission "
                    "to access this storage object"
                ),
            )

    # ------------------------------------------------------------------
    # Signed URLs
    # ------------------------------------------------------------------

    def generate_signed_url(
        self,
        image_path: str,
        expiration_seconds: Optional[int] = None,
    ) -> str:
        """
        Generate a short-lived signed read URL.

        The object is retrieved first so existence is verified.
        The URL itself is never logged.
        """

        blob = self._get_blob(
            image_path
        )

        exp_seconds = (
            expiration_seconds
            if expiration_seconds is not None
            else self.settings.signed_url_expiration_seconds
        )

        if (
            exp_seconds <= 0
            or exp_seconds > 3600
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid signed URL expiration",
            )

        try:
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=datetime.timedelta(
                    seconds=exp_seconds
                ),
                method="GET",
            )

            logger.info(
                "Generated temporary image capability"
            )

            return signed_url

        except Exception as exc:
            logger.error(
                "Failed to generate signed URL: %s",
                type(exc).__name__,
            )

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "Failed to generate secure temporary "
                    "access for image"
                ),
            ) from exc

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_image_metadata(
        self,
        image_path: str,
    ) -> Dict[str, Any]:
        """
        Return the actual trusted Storage metadata for an image.

        This is intentionally based on get_blob(), not bucket.blob(),
        because normalize_storage_metadata() needs the object's loaded
        custom metadata.
        """

        clean_path = self.validate_image_path(
            image_path
        )

        blob = self._get_blob(
            clean_path
        )

        try:
            metadata = normalize_storage_metadata(
                clean_path,
                blob,
            )

            logger.info(
                "Resolved Storage metadata for image object"
            )

            return metadata

        except Exception as exc:
            logger.error(
                "Failed to normalize image metadata: %s",
                type(exc).__name__,
            )

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to resolve image metadata",
            ) from exc


# ============================================================================
# Singleton
# ============================================================================

_storage_service_instance: Optional[
    FirebaseStorageService
] = None


def get_storage_service() -> FirebaseStorageService:
    global _storage_service_instance

    if _storage_service_instance is None:
        _storage_service_instance = (
            FirebaseStorageService()
        )

    return _storage_service_instance


# ============================================================================
# Firebase Authentication
# ============================================================================

def verify_firebase_token(
    token: str,
) -> Dict[str, Any]:
    """Verify Firebase ID token and return decoded claims."""

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Authentication credentials were not provided"
            ),
            headers={
                "WWW-Authenticate": "Bearer"
            },
        )

    try:
        if not firebase_admin._apps:
            FirebaseStorageService()

        payload = auth.verify_id_token(
            token
        )

        user_id = payload.get(
            "uid"
        )

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=(
                    "Invalid Firebase authentication token"
                ),
                headers={
                    "WWW-Authenticate": "Bearer"
                },
            )

        return payload

    except HTTPException:
        raise

    except Exception as exc:
        logger.warning(
            "Firebase token verification failed: %s",
            type(exc).__name__,
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Invalid or expired Firebase credentials"
            ),
            headers={
                "WWW-Authenticate": "Bearer"
            },
        ) from exc