import logging
import mimetypes
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Security,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import Settings, get_settings

# Cognito handles all API authentication.
from services.cognito_auth import verify_cognito_token

# S3 presigned URLs for browser uploads and analysis reads.
from services.s3_service import S3Service, get_s3_service

from services.qwen_service import (
    QwenService,
    get_qwen_service,
)
from services.input_manifest import (
    InputManifestCompatibilityError,
)

from api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    StatusResponse,
)


logger = logging.getLogger("akasha.api")

router = APIRouter()

security = HTTPBearer(auto_error=False)

MAX_PHYSICAL_FILES = 4
MAX_UPLOAD_SIZE_BYTES = 512 * 1024 * 1024
PRESIGNED_UPLOAD_EXPIRES_SECONDS = 900
ALLOWED_UPLOAD_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/tiff",
    "image/geotiff",
    "application/octet-stream",
}


class UploadUrlRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    content_type: str = Field(min_length=1, max_length=128)
    size: int = Field(gt=0, le=MAX_UPLOAD_SIZE_BYTES)


class UploadUrlResponse(BaseModel):
    upload_url: str
    object_key: str
    image_id: str
    expires_in: int


# ---------------------------------------------------------------------------
# Authentication helper
# ---------------------------------------------------------------------------


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(
        security
    ),
) -> Dict[str, Any]:
    """
    Authenticate an API request using a Cognito JWT.

    Expected header:

        Authorization: Bearer <Cognito access token>

    The token is verified against the configured Cognito User Pool,
    including its signature, issuer, expiry, and application binding.
    """

    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return verify_cognito_token(credentials.credentials)


# ---------------------------------------------------------------------------
# Auth introspection
# ---------------------------------------------------------------------------


@router.get("/auth/me")
def me(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return the authenticated Cognito user's identity."""

    user_id = (
        current_user.get("sub")
        or current_user.get("uid")
        or current_user.get("user_id")
    )

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authenticated user",
        )

    return {
        "id": str(user_id),
        "email": current_user.get("email", ""),
        "token_use": current_user.get("token_use"),
    }


@router.get("/auth/test")
def auth_test(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Lightweight Cognito authentication check endpoint."""

    user_id = (
        current_user.get("sub")
        or current_user.get("uid")
        or current_user.get("user_id")
    )

    return {
        "authenticated": True,
        "user_id": str(user_id) if user_id else None,
        "email": current_user.get("email"),
        "token_use": current_user.get("token_use"),
    }


# ---------------------------------------------------------------------------
# S3 browser uploads
# ---------------------------------------------------------------------------


@router.post(
    "/storage/upload-url",
    response_model=UploadUrlResponse,
)
def create_upload_url(
    request: UploadUrlRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    s3_service: S3Service = Depends(get_s3_service),
) -> UploadUrlResponse:
    """
    Create a short-lived S3 presigned PUT URL for the authenticated user.

    The frontend never chooses the final S3 object key. The backend creates
    a user-scoped key so an authenticated client cannot write into another
    user's namespace. The actual file bytes are uploaded directly by the
    browser to S3 using the returned presigned URL.
    """

    user_id = (
        current_user.get("sub")
        or current_user.get("uid")
        or current_user.get("user_id")
        or current_user.get("id")
    )

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authenticated user identity",
        )

    filename = Path(request.filename).name.strip()
    if not filename or filename in {".", ".."}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A valid filename is required.",
        )

    # Keep filenames readable while removing path/control characters.
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename.",
        )

    content_type = request.content_type.strip().lower()
    if content_type not in ALLOWED_UPLOAD_CONTENT_TYPES:
        guessed_type, _ = mimetypes.guess_type(filename)
        if guessed_type not in ALLOWED_UPLOAD_CONTENT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=(
                    "Unsupported image content type. "
                    "Use PNG, JPEG, TIFF/GeoTIFF, or octet-stream."
                ),
            )
        content_type = guessed_type

    image_id = str(uuid4())
    object_key = f"users/{user_id}/imagery/{image_id}/{filename}"

    try:
        upload_url = s3_service.generate_upload_url(
            object_key=object_key,
            content_type=content_type,
            expiration_seconds=PRESIGNED_UPLOAD_EXPIRES_SECONDS,
        )
    except Exception as exc:
        logger.exception(
            "Failed to create S3 presigned upload URL for user=%s",
            user_id,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to prepare the imagery upload. Please try again later.",
        ) from exc

    return UploadUrlResponse(
        upload_url=upload_url,
        object_key=object_key,
        image_id=image_id,
        expires_in=PRESIGNED_UPLOAD_EXPIRES_SECONDS,
    )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
)
async def analyze_image(
    request: AnalyzeRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    s3_service: S3Service = Depends(get_s3_service),
    qwen_service: QwenService = Depends(get_qwen_service),
    settings: Settings = Depends(get_settings),
) -> AnalyzeResponse:
    """
    Production AI analysis endpoint.

    Authenticated users submit one to four S3 object keys (previously
    uploaded via /storage/upload-url) along with a natural-language query.

    Flow:
      1. Verify Cognito JWT → extract sub (user_id)
      2. Resolve image paths from image_paths or fallback image_path
      3. Validate S3 object keys (path traversal protection)
      4. Enforce user-scoped namespace: users/{user_id}/imagery/
      5. HEAD each S3 object to verify existence + metadata
      6. Build trusted metadata merging S3 HEAD with client hints
      7. Validate input_manifest from request
      8. Generate S3 presigned GET URLs
      9. Invoke QwenService.analyze()
     10. Return AnalyzeResponse(answer=...)
    """

    # -----------------------------------------------------------------------
    # 1. Extract authenticated user identity
    # -----------------------------------------------------------------------

    user_id = (
        current_user.get("sub")
        or current_user.get("uid")
        or current_user.get("user_id")
        or current_user.get("id")
    )

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authenticated user identity",
        )

    # -----------------------------------------------------------------------
    # 2. Resolve image paths
    # -----------------------------------------------------------------------

    raw_paths: List[str] = request.image_paths or []

    if not raw_paths and request.image_path:
        raw_paths = [request.image_path]

    if not raw_paths:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one image path is required.",
        )

    if len(raw_paths) > MAX_PHYSICAL_FILES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"A maximum of {MAX_PHYSICAL_FILES} "
                "physical image files is supported."
            ),
        )

    # -----------------------------------------------------------------------
    # 3. Validate S3 object keys
    # -----------------------------------------------------------------------

    clean_paths: List[str] = []

    for raw_path in raw_paths:
        try:
            clean_path = s3_service.validate_object_key(raw_path)
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image path: {raw_path!r}",
            )

        clean_paths.append(clean_path)

    # -----------------------------------------------------------------------
    # 4. Enforce user-scoped S3 namespace
    # -----------------------------------------------------------------------

    expected_prefix = f"users/{user_id}/imagery/"

    for clean_path in clean_paths:
        if not clean_path.startswith(expected_prefix):
            logger.warning(
                "S3 namespace violation: user=%s path=%s",
                user_id,
                clean_path,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Access denied: you do not have permission "
                    "to access this storage object."
                ),
            )

    # -----------------------------------------------------------------------
    # 5. HEAD each S3 object (verify existence + metadata)
    # -----------------------------------------------------------------------

    s3_metadata_list: List[Dict[str, Any]] = []

    for clean_path in clean_paths:
        try:
            meta = s3_service.head_object(clean_path)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Could not verify imagery in storage.",
            ) from exc

        s3_metadata_list.append(meta)

    # -----------------------------------------------------------------------
    # 6. Build trusted file metadata
    # -----------------------------------------------------------------------

    client_metadata_list: List[Dict[str, Any]] = (
        request.image_metadata or []
    )

    file_records: List[Dict[str, Any]] = []

    for i, (clean_path, s3_meta) in enumerate(
        zip(clean_paths, s3_metadata_list)
    ):
        client_hint = (
            client_metadata_list[i]
            if i < len(client_metadata_list)
            else {}
        )

        filename = (
            client_hint.get("filename")
            or Path(clean_path).name
        )

        modality = client_hint.get("modality")
        if not modality:
            lower_name = filename.lower()
            if any(k in lower_name for k in ["sentinel-1", "sentinel_1", "sar", "sigma0", "gamma0"]) or re.search(r"(?:^|[\-_])(vv|vh)(?=[_.-]|$)", lower_name):
                modality = "sar"
            else:
                modality = "optical"

        polarization = client_hint.get("polarization")
        if not polarization and modality == "sar":
            m_pol = re.findall(r"(?:^|[\-_])(vv|vh)(?=[_.-]|$)", filename.lower())
            if m_pol:
                polarization = m_pol[-1].upper()

        acquisition_time = client_hint.get("acquisition_time")
        if not acquisition_time:
            m_time = re.search(r"(20\d{2})[-_](\d{2})[-_](\d{2})", filename)
            if m_time:
                acquisition_time = f"{m_time.group(1)}-{m_time.group(2)}-{m_time.group(3)}"

        record: Dict[str, Any] = {
            "object_key": clean_path,
            "content_type": (
                s3_meta.get("content_type")
                or client_hint.get("content_type")
            ),
            "size": (
                s3_meta.get("size")
                or client_hint.get("size")
            ),
            "filename": filename,
            "modality": modality,
            "polarization": polarization,
            "acquisition_time": acquisition_time,
            "observation_id": client_hint.get("observation_id"),
            "relationship_type": client_hint.get("relationship_type"),
            "spatially_corresponding": client_hint.get("spatially_corresponding"),
            "co_registered": client_hint.get("co_registered"),
            "same_geographic_area": client_hint.get("same_geographic_area"),
        }

        file_records.append(record)

    # -----------------------------------------------------------------------
    # 7. Validate input_manifest
    # -----------------------------------------------------------------------

    from services.input_manifest import build_input_manifest

    manifest_data = request.input_manifest or request.manifest

    try:
        trusted_manifest = build_input_manifest(file_records)
    except InputManifestCompatibilityError as exc:
        if isinstance(manifest_data, dict) and "observations" in manifest_data:
            trusted_manifest = manifest_data
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            )
    except Exception as exc:
        logger.exception("Input manifest construction failed")
        if isinstance(manifest_data, dict) and "observations" in manifest_data:
            trusted_manifest = manifest_data
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not build a valid input manifest from the provided imagery.",
            )

    # -----------------------------------------------------------------------
    # 8. Generate S3 presigned GET URLs
    # -----------------------------------------------------------------------

    signed_urls: List[str] = []

    try:
        for clean_path in clean_paths:
            signed_urls.append(
                s3_service.generate_download_url(
                    object_key=clean_path,
                    expiration_seconds=settings.signed_url_expiration_seconds,
                )
            )
    except Exception as exc:
        logger.exception(
            "Failed to generate signed GET URLs for user=%s",
            user_id,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to generate secure access for imagery.",
        )

    # -----------------------------------------------------------------------
    # 9. Invoke Qwen
    # -----------------------------------------------------------------------

    from starlette.concurrency import run_in_threadpool

    try:
        answer = await run_in_threadpool(
            qwen_service.analyze,
            user_message=request.query,
            user_request=request.query,
            image_url=signed_urls[0] if signed_urls else "",
            signed_urls=signed_urls,
            image_urls=signed_urls,
            max_new_tokens=request.max_new_tokens,
            manifest=trusted_manifest,
            input_manifest=trusted_manifest,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Qwen request failed for user=%s",
            user_id,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "AI analysis is temporarily unavailable. "
                "Please try again later."
            ),
        ) from exc

    # -----------------------------------------------------------------------
    # 10. Return answer
    # -----------------------------------------------------------------------

    logger.info(
        "Qwen request completed: user=%s physical_files=%d",
        user_id,
        len(clean_paths),
    )

    return AnalyzeResponse(answer=answer)


# ---------------------------------------------------------------------------
# Health / status
# ---------------------------------------------------------------------------


@router.get(
    "/status",
    response_model=StatusResponse,
)
def get_status(
    settings: Settings = Depends(get_settings),
) -> StatusResponse:
    """
    Lightweight health/configuration check.

    Does NOT perform model inference.
    """

    return StatusResponse(
        status="AKASHA API running",
        version="1.0.0",
        qwen_configured=settings.is_qwen_configured,
        aws_configured=settings.is_aws_configured,
    )
