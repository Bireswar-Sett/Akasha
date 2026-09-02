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

from api.schemas import AnalyzeRequest, AnalyzeResponse, StatusResponse
from config import Settings, get_settings
from services.firebase_service import (
    FirebaseStorageService,
    get_storage_service,
    verify_firebase_token,
)
from services.qwen import query_qwen
from services.qwen_service import QwenService, get_qwen_service

logger = logging.getLogger("akasha.api")
router = APIRouter()

security = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> Dict[str, Any]:
    """
    Authenticate the current user using Firebase ID token in Authorization header.
    Expects: 'Bearer <firebase_id_token>'
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    decoded_token = verify_firebase_token(token)
    return decoded_token


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
    # Safe structured logging - image path is logged, but NEVER any signed URLs or secrets
    logger.info(f"Qwen request started for image_path={request.image_path}")

    user_id = current_user.get("uid")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user identity lacks a valid UID",
        )

    # 1. Validate image path format and security
    clean_path = storage_service.validate_image_path(request.image_path)

    # 2. Verify authorization
    storage_service.verify_user_authorization(user_id=user_id, image_path=clean_path)

    # 3. Generate short-lived signed URL (verifies object exists first)
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
):
    """
    Direct multipart query endpoint (retained for backward compatibility).
    """
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
