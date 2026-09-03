import logging
from typing import Dict, Any, Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Security,
    status,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import Settings, get_settings
from services.firebase_service import (
    FirebaseStorageService,
    get_storage_service,
    verify_firebase_token,
)
from services.qwen_service import QwenService, get_qwen_service

from api.schemas import (AnalyzeRequest, AnalyzeResponse, StatusResponse,
                         RegisterRequest, LoginRequest, TokenResponse, UserResponse)
from config import Settings, get_settings
from services.firebase_service import FirebaseStorageService, get_storage_service
from services.qwen_service import QwenService, get_qwen_service
from services.auth_service import register_user, authenticate_user, create_access_token, decode_access_token

logger = logging.getLogger("akasha.api")
router = APIRouter()

security = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> Dict[str, Any]:

    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return decode_access_token(credentials.credentials)


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

    return UserResponse(
        id=current_user["id"],
        email=current_user["email"],
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    request: AnalyzeRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    storage_service: FirebaseStorageService = Depends(get_storage_service),
    qwen_service: QwenService = Depends(get_qwen_service),
) -> AnalyzeResponse:
    """
    Satellite image analysis endpoint.

    Flow:
    1. Authenticate via Firebase ID token.
    2. Validate Firebase Storage image path.
    3. Verify user owns the storage object.
    4. Generate short-lived signed URL (internal only, never returned).
    5. Call Qwen Gradio Space /ask_akasha with signed URL.
    6. Return synthesized answer.
    """
    logger.info(f"Qwen request started for image_path={request.image_path}")

    user_id = current_user.get("id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user identity lacks a valid ID",
        )

    clean_path = storage_service.validate_image_path(request.image_path)
    storage_service.verify_user_authorization(user_id=user_id, image_path=clean_path)

    signed_url = storage_service.generate_signed_url(clean_path)

    logger.info("Calling Qwen Space")
    answer = qwen_service.analyze(
        user_message=request.query,
        image_url=signed_url,
        max_new_tokens=request.max_new_tokens,
    )

    logger.info("Qwen request completed")
    return AnalyzeResponse(answer=answer)


@router.get("/status", response_model=StatusResponse)
def get_status(settings: Settings = Depends(get_settings)) -> StatusResponse:
    """
    Lightweight health and configuration check.
    Does NOT perform model inference.
    """
    return StatusResponse(
        status="AKASHA API running",
        version="1.0.0",
        qwen_configured=settings.is_qwen_configured,
        firebase_configured=settings.is_firebase_configured,
    )
