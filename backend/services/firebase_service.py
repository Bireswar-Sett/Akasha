import os
import re
import datetime
import logging
from typing import Optional, Dict, Any

from fastapi import HTTPException, status
import firebase_admin
from firebase_admin import auth, credentials, storage
from services.auth_service import register_user, authenticate_user, create_access_token, decode_access_token

from config import get_settings

logger = logging.getLogger("akasha.firebase")


class FirebaseStorageService:
    def __init__(self):
        self.settings = get_settings()
        self._init_firebase()

    def _init_firebase(self) -> None:
        """Idempotently initialize Firebase Admin SDK."""
        if not firebase_admin._apps:
            key_path = self.settings.firebase_service_account_key_path
            bucket_name = self.settings.firebase_storage_bucket
            project_id = self.settings.firebase_project_id

            options = {}
            if bucket_name:
                options["storageBucket"] = bucket_name
            if project_id:
                options["projectId"] = project_id

            try:
                if key_path and os.path.isfile(key_path):
                    cred = credentials.Certificate(key_path)
                    firebase_admin.initialize_app(cred, options)
                    logger.info("Firebase Admin initialized with service account key")
                elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                    cred = credentials.ApplicationDefault()
                    firebase_admin.initialize_app(cred, options)
                    logger.info("Firebase Admin initialized with Application Default Credentials")
                else:
                    # Fallback initialization (e.g., in testing or emulator environments)
                    firebase_admin.initialize_app(options=options)
                    logger.info("Firebase Admin initialized with default options")
            except Exception as e:
                logger.warning(f"Firebase Admin initialization deferred or failed: {e}")

    def get_bucket(self):
        """Retrieve default storage bucket."""
        try:
            return storage.bucket(self.settings.firebase_storage_bucket)
        except Exception as e:
            logger.error("Failed to acquire Firebase storage bucket")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Storage service unavailable"
            )

    def validate_image_path(self, image_path: str) -> str:
        """
        Validate path format and guard against path traversal.
        Allowed patterns: 'users/<uid>/...' or 'satellite_images/<uid>/...'
        """
        if not image_path or not isinstance(image_path, str):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image path must be a non-empty string"
            )

        clean_path = image_path.strip().replace("\\", "/")

        # Prevent path traversal attacks
        if ".." in clean_path or clean_path.startswith("/") or "\0" in clean_path:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image path: directory traversal or illegal characters detected"
            )

        # Ensure path has valid file extension
        valid_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")
        if not any(clean_path.lower().endswith(ext) for ext in valid_extensions):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image path: unsupported image file format"
            )

        return clean_path

    def verify_user_authorization(self, user_id: str, image_path: str) -> None:
        """
        Validate that the requested Firebase Storage object belongs to / is accessible by the user.
        Conforms to project structure:
          - users/{user_id}/imagery/...
          - users/{user_id}/...
          - satellite_images/{user_id}/...
        """
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User identity could not be verified"
            )

        parts = image_path.strip("/").split("/")
        if len(parts) < 3:
            # Need at least prefix/user_id/filename
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image path format. Expected '<category>/<user_id>/<filename>'"
            )

        prefix, path_user_id = parts[0], parts[1]

        # Valid allowed prefixes in project architecture
        allowed_prefixes = {"users", "satellite_images"}
        if prefix not in allowed_prefixes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: unauthorized storage namespace"
            )

        if path_user_id != user_id:
            logger.warning(f"Authorization failure: user={user_id} attempted access to path belonging to {path_user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: you do not have permission to access this storage object"
            )

    def generate_signed_url(self, image_path: str, expiration_seconds: Optional[int] = None) -> str:
        """
        Generates a SHORT-LIVED signed read URL for the requested Firebase Storage object.
        Verifies object existence first.
        NEVER logs or returns the signed URL.
        """
        clean_path = self.validate_image_path(image_path)
        bucket = self.get_bucket()
        blob = bucket.blob(clean_path)

        # Verify object existence in bucket
        try:
            if not blob.exists():
                logger.warning(f"Storage object not found: image_path={clean_path}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="The requested image object was not found in storage"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error checking storage object existence: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to verify storage object existence"
            )

        exp_seconds = expiration_seconds or self.settings.signed_url_expiration_seconds
        try:
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=datetime.timedelta(seconds=exp_seconds),
                method="GET"
            )
            logger.info("Generated temporary image capability")
            return signed_url
        except Exception as e:
            logger.error(f"Failed to generate signed URL: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate secure temporary access for image"
            )


# Singleton instance
_storage_service_instance: Optional[FirebaseStorageService] = None


def get_storage_service() -> FirebaseStorageService:
    global _storage_service_instance
    if _storage_service_instance is None:
        _storage_service_instance = FirebaseStorageService()
    return _storage_service_instance


def verify_firebase_token(token: str) -> Dict[str, Any]:
    """Verify Firebase ID token and return decoded token dict."""
    try:
        payload = decode_access_token(credentials.credentials)

        user_id = payload.get("sub")

        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token"
            )

        return get_user_by_id(user_id)
    except Exception as e:
        logger.warning(f"Firebase token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication credentials"
        )
