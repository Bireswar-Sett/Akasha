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
    user_id = current_user.get("uid") or current_user.get("id") or current_user.get("user_id")
    email = current_user.get("email") or ""
    return UserResponse(
        id=str(user_id),
        email=email,
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    request: AnalyzeRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    storage_service: FirebaseStorageService = Depends(get_storage_service),
    qwen_service: QwenService = Depends(get_qwen_service),
) -> AnalyzeResponse:
    """
    Production AI analysis endpoint:
    1. Authenticate the request via Firebase Auth.
    2. Validate request payload and storage path syntax.
    3. Verify user authorization to access the specified Firebase Storage object.
    4. Generate a short-lived signed read URL internally.
    5. Dispatch inference to Qwen Hugging Face Gradio Space.
    6. Return Qwen's final synthesized response without exposing internal signed URLs.
    """

    # Safe structured logging - never log signed URLs or secrets
    logger.info(
        "Qwen request started for image_path=%s",
        request.image_path,
    )

    user_id = current_user.get("uid")

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Firebase user identity",
        )

    # 1. Validate image path
    clean_path = storage_service.validate_image_path(request.image_path)

    # 2. Verify authorization
    storage_service.verify_user_authorization(
        user_id=user_id,
        image_path=clean_path,
    )

    # 3. Generate short-lived signed URL
    signed_url = storage_service.generate_signed_url(clean_path)

    # 4. Call Qwen Gradio Space
    logger.info("Calling Qwen Space")

    answer = qwen_service.analyze(
        user_message=request.query,
        image_url=signed_url,
        max_new_tokens=request.max_new_tokens,
    )

    logger.info("Qwen request completed")

    return AnalyzeResponse(answer=answer)


@router.post("/query")
async def query(
    query: str = Form(default=""),
    images: List[UploadFile] = File(default=[]),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Direct multipart query endpoint kept behind Firebase auth."""
    image_data_list = []
    for img in images:
        content = await img.read()
        if content:
            image_data_list.append({
                "filename": img.filename or "image.png",
                "content_type": img.content_type or "image/png",
                "bytes": content,
            })

    return query_qwen(query, image_data_list)


@router.get("/status", response_model=StatusResponse)
def get_status(settings: Settings = Depends(get_settings)) -> StatusResponse:
    """
    Lightweight health and configuration check.
    Does NOT perform model inference to preserve ZeroGPU quota.
    """
    return StatusResponse(
        status="AKASHA API running",
        version="1.0.0",
        qwen_configured=settings.is_qwen_configured,
        firebase_configured=settings.is_firebase_configured,
    )
