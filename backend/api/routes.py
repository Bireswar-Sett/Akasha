import logging
from typing import List, Dict, Any, Optional

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
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import Settings, get_settings
from services.firebase_service import (
    FirebaseStorageService,
    get_storage_service,
    verify_firebase_token,
)
from services.qwen import query_qwen
from services.qwen_service import QwenService, get_qwen_service
from services.input_manifest import (
    InputManifestCompatibilityError,
    build_input_manifest,
)
from services.auth_service import (
    register_user,
    authenticate_user,
    create_access_token,
)

from api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    StatusResponse,
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    UserResponse,
)

logger = logging.getLogger("akasha.api")
router = APIRouter()

security = HTTPBearer(auto_error=False)

MAX_PHYSICAL_FILES = 4


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> Dict[str, Any]:
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return verify_firebase_token(credentials.credentials)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


@router.post(
    "/auth/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(request: RegisterRequest) -> UserResponse:

    user = register_user(
        request.email,
        request.password,
    )

    return UserResponse(
        id=user["id"],
        email=user["email"],
    )


@router.post(
    "/auth/login",
    response_model=TokenResponse,
)
def login(request: LoginRequest) -> TokenResponse:

    user = authenticate_user(
        request.email,
        request.password,
    )

    token = create_access_token(user["id"])

    return TokenResponse(
        access_token=token,
        token_type="bearer",
    )


@router.get(
    "/auth/me",
    response_model=UserResponse,
)
def me(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> UserResponse:

    user_id = (
        current_user.get("uid")
        or current_user.get("id")
        or current_user.get("user_id")
    )

    email = current_user.get("email") or ""

    return UserResponse(
        id=str(user_id),
        email=email,
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
    storage_service: FirebaseStorageService = Depends(
        get_storage_service
    ),
    qwen_service: QwenService = Depends(
        get_qwen_service
    ),
) -> AnalyzeResponse:
    """
    Production AI analysis endpoint.

    Flow:

        Firebase-authenticated frontend request
                    ↓
        validate physical file list
                    ↓
        verify user access to every Storage object
                    ↓
        retrieve trusted Storage metadata
                    ↓
        build logical observation manifest
                    ↓
        generate short-lived signed URLs
                    ↓
        invoke Qwen controller
                    ↓
        return final synthesized answer

    Important:
        VV + VH are TWO physical files but ONE logical SAR observation.

    The frontend contract remains:

        query
        image_path
        image_paths
        max_new_tokens
    """

    # -----------------------------------------------------------------------
    # 1. Resolve authenticated user
    # -----------------------------------------------------------------------

    user_id = (
        current_user.get("uid")
        or current_user.get("user_id")
        or current_user.get("id")
    )

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Firebase user identity",
        )

    # -----------------------------------------------------------------------
    # 2. Resolve physical Storage paths
    #
    # image_paths is the preferred field.
    # image_path remains supported for backwards compatibility.
    # -----------------------------------------------------------------------

    raw_paths = request.image_paths

    if not raw_paths:
        if request.image_path:
            raw_paths = [request.image_path]
        else:
            raw_paths = []

    # Remove accidental empty values while preserving order.
    image_paths = [
        path.strip()
        for path in raw_paths
        if isinstance(path, str) and path.strip()
    ]

    if not image_paths:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one satellite image must be provided.",
        )

    if len(image_paths) > MAX_PHYSICAL_FILES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"A maximum of {MAX_PHYSICAL_FILES} physical image files "
                "can be analyzed in one request."
            ),
        )

    # -----------------------------------------------------------------------
    # 3. Validate + authorize every physical file
    #
    # Never trust the frontend to decide whether a Storage path belongs
    # to the authenticated user.
    # -----------------------------------------------------------------------

    clean_paths: list[str] = []

    for image_path in image_paths:
        try:
            clean_path = (
                storage_service.validate_image_path(
                    image_path
                )
            )

            storage_service.verify_user_authorization(
                user_id=user_id,
                image_path=clean_path,
            )

            clean_paths.append(clean_path)

        except HTTPException:
            raise

        except Exception as exc:
            logger.exception(
                "Storage authorization failed for one requested object"
            )

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="One or more requested image files are invalid or inaccessible.",
            ) from exc

    # -----------------------------------------------------------------------
    # 4. Retrieve TRUSTED Firebase Storage metadata
    #
    # This is the only metadata source used for manifest construction.
    # Do not infer SAR/optical/VV/VH from filenames here.
    # -----------------------------------------------------------------------

    try:
        trusted_metadata = [
            storage_service.get_image_metadata(
                path
            )
            for path in clean_paths
        ]

    except Exception as exc:
        logger.exception(
            "Failed to retrieve trusted Storage metadata"
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "Unable to retrieve trusted imagery metadata. "
                "Please try again later."
            ),
        ) from exc

    # -----------------------------------------------------------------------
    # 5. Build logical observation manifest
    #
    # Examples:
    #
    #   optical.tiff
    #       → one optical observation
    #
    #   VV.tiff + VH.tiff
    #       → one SAR observation
    #
    #   optical.tiff + VV.tiff + VH.tiff
    #       → optical observation + SAR observation
    #
    #   VV1 + VH1 + VV2 + VH2
    #       → two SAR observations
    # -----------------------------------------------------------------------

    try:
        manifest = build_input_manifest(
            trusted_metadata
        )

    except InputManifestCompatibilityError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unexpected input-manifest construction failure"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "The uploaded imagery could not be converted into "
                "a valid analysis manifest."
            ),
        ) from exc

    # -----------------------------------------------------------------------
    # 6. Generate short-lived signed URLs
    #
    # Generate these only AFTER authorization and manifest validation.
    # -----------------------------------------------------------------------

    signed_urls: list[str] = []

    try:
        for clean_path in clean_paths:
            signed_urls.append(
                storage_service.generate_signed_url(
                    clean_path
                )
            )

    except Exception as exc:
        logger.exception(
            "Failed to generate signed image URLs"
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "Unable to prepare the imagery for AI analysis. "
                "Please try again later."
            ),
        ) from exc

    # -----------------------------------------------------------------------
    # 7. Log safe request metadata
    #
    # Do NOT log:
    #   - signed URLs
    #   - Firebase paths
    #   - credentials
    #
    # Only log non-sensitive structural information.
    # -----------------------------------------------------------------------

    logger.info(
        "Qwen request started: user=%s physical_files=%d observations=%d",
        user_id,
        len(clean_paths),
        len(manifest.get("observations", [])),
    )

    # -----------------------------------------------------------------------
    # 8. Invoke Qwen
    #
    # Keep the transport contract stable.
    #
    # image_url  = first physical image
    # image_urls = all physical images
    #
    # Qwen receives the logical manifest and trusted metadata as context.
    # -----------------------------------------------------------------------

    qwen_kwargs = {
        "user_message": request.query,
        "image_url": signed_urls[0],
        "max_new_tokens": request.max_new_tokens,
        "manifest": manifest,
        "image_metadata": trusted_metadata,
    }

    if len(signed_urls) > 1:
        qwen_kwargs["image_urls"] = signed_urls

    try:
        answer = qwen_service.analyze(
            **qwen_kwargs
        )

    except HTTPException:
        raise

    except TimeoutError as exc:
        logger.exception(
            "Qwen request timed out"
        )

        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="AI analysis timed out. Please try again.",
        ) from exc

    except Exception as exc:
        logger.exception(
            "Qwen request failed"
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "AI analysis is temporarily unavailable. "
                "Please try again later."
            ),
        ) from exc

    # -----------------------------------------------------------------------
    # 9. Return only the final answer
    #
    # Signed URLs and internal manifests are never returned to the frontend
    # through this response model unless AnalyzeResponse explicitly exposes
    # them elsewhere.
    # -----------------------------------------------------------------------

    logger.info(
        "Qwen request completed: user=%s physical_files=%d",
        user_id,
        len(clean_paths),
    )

    return AnalyzeResponse(
        answer=answer
    )


# ---------------------------------------------------------------------------
# Legacy/direct multipart endpoint
# ---------------------------------------------------------------------------


@router.post("/query")
async def query(
    query: str = Form(default=""),
    images: List[UploadFile] = File(default=[]),
    current_user: Dict[str, Any] = Depends(
        get_current_user
    ),
):
    """
    Direct multipart query endpoint kept behind Firebase authentication.

    This endpoint is intentionally separate from the production Storage-based
    `/analyze` workflow.
    """

    if len(images) > MAX_PHYSICAL_FILES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"A maximum of {MAX_PHYSICAL_FILES} "
                "physical image files is supported."
            ),
        )

    image_data_list = []

    for img in images:
        content = await img.read()

        if content:
            image_data_list.append(
                {
                    "filename": (
                        img.filename
                        or "image.png"
                    ),
                    "content_type": (
                        img.content_type
                        or "application/octet-stream"
                    ),
                    "bytes": content,
                }
            )

    return query_qwen(
        query,
        image_data_list,
    )


# ---------------------------------------------------------------------------
# Health/status
# ---------------------------------------------------------------------------


@router.get(
    "/status",
    response_model=StatusResponse,
)
def get_status(
    settings: Settings = Depends(
        get_settings
    ),
) -> StatusResponse:
    """
    Lightweight health/configuration check.

    Does NOT perform model inference and therefore does not consume ZeroGPU
    inference resources.
    """

    return StatusResponse(
        status="AKASHA API running",
        version="1.0.0",
        qwen_configured=settings.is_qwen_configured,
        firebase_configured=settings.is_firebase_configured,
    )