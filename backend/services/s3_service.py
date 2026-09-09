from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from fastapi import HTTPException, status

from config import get_settings


logger = logging.getLogger("akasha.s3")


class S3Service:
    """
    AWS S3 service using the ECS task IAM role.

    Production flow:

        ECS task
             ↓
        AkashaECSTaskRole
             ↓
        S3
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def _get_s3_client(self):
        """
        Return an S3 client using the AWS credentials provided by the
        ECS task role through the standard boto3 credential chain.
        """
        return boto3.client(
            "s3",
            region_name=self.settings.aws_region,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "virtual"},
            ),
        )

    # ------------------------------------------------------------------
    # Object path validation
    # ------------------------------------------------------------------

    def validate_object_key(
        self,
        object_key: str,
    ) -> str:
        if (
            not object_key
            or not isinstance(object_key, str)
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="S3 object key must be a non-empty string",
            )

        clean_key = (
            object_key
            .strip()
            .replace("\\", "/")
        )

        if (
            clean_key.startswith("/")
            or ".." in clean_key
            or "\0" in clean_key
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid S3 object key",
            )

        return clean_key

    # ------------------------------------------------------------------
    # Upload URLs
    # ------------------------------------------------------------------

    def generate_upload_url(
        self,
        object_key: str,
        content_type: str,
        expiration_seconds: int = 900,
    ) -> str:
        """
        Generate a temporary PUT URL for one S3 object.
        """

        object_key = self.validate_object_key(
            object_key
        )

        if not content_type:
            content_type = (
                "application/octet-stream"
            )

        if not (
            1 <= expiration_seconds <= 900
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Upload URL expiration must be "
                    "between 1 and 900 seconds"
                ),
            )

        try:
            client = self._get_s3_client()

            return client.generate_presigned_url(
                ClientMethod="put_object",
                Params={
                    "Bucket": self.settings.aws_s3_bucket,
                    "Key": object_key,
                    "ContentType": content_type,
                },
                ExpiresIn=expiration_seconds,
                HttpMethod="PUT",
            )

        except ClientError as exc:
            logger.exception(
                "Failed to generate S3 upload URL"
            )

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unable to prepare S3 upload",
            ) from exc

    # ------------------------------------------------------------------
    # Read URLs
    # ------------------------------------------------------------------

    def generate_download_url(
        self,
        object_key: str,
        expiration_seconds: int = 1800,
    ) -> str:
        """
        Generate a temporary GET URL for an S3 object.
        """

        object_key = self.validate_object_key(
            object_key
        )

        if not (
            1 <= expiration_seconds <= 3600
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Download URL expiration must be "
                    "between 1 and 3600 seconds"
                ),
            )

        try:
            client = self._get_s3_client()

            return client.generate_presigned_url(
                ClientMethod="get_object",
                Params={
                    "Bucket": self.settings.aws_s3_bucket,
                    "Key": object_key,
                },
                ExpiresIn=expiration_seconds,
                HttpMethod="GET",
            )

        except ClientError as exc:
            logger.exception(
                "Failed to generate S3 download URL"
            )

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unable to prepare secure image access",
            ) from exc

    # ------------------------------------------------------------------
    # Object metadata
    # ------------------------------------------------------------------

    def head_object(
        self,
        object_key: str,
    ) -> Dict[str, Any]:
        object_key = self.validate_object_key(
            object_key
        )

        try:
            client = self._get_s3_client()

            response = client.head_object(
                Bucket=self.settings.aws_s3_bucket,
                Key=object_key,
            )

            return response

        except ClientError as exc:
            error_code = (
                exc.response
                .get("Error", {})
                .get("Code")
            )

            if error_code in {
                "404",
                "NoSuchKey",
                "NotFound",
            }:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="S3 object not found",
                ) from exc

            logger.exception(
                "Failed to inspect S3 object"
            )

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unable to access S3 object",
            ) from exc


_s3_service_instance: Optional[S3Service] = None


def get_s3_service() -> S3Service:
    global _s3_service_instance

    if _s3_service_instance is None:
        _s3_service_instance = S3Service()

    return _s3_service_instance