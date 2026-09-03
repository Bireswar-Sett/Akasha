from typing import Optional
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    image_path: str = Field(
        ...,
        min_length=1,
        description="Firebase Storage object path (e.g., users/{uid}/imagery/{file} or satellite_images/{uid}/{file})",
        examples=["satellite_images/user123/image_001.png"]
    )
    query: str = Field(
        ...,
        min_length=1,
        description="User question or analysis instructions for the satellite imagery",
        examples=["Analyze this satellite image. Describe the visible features, especially buildings."]
    )
    max_new_tokens: int = Field(
        default=256,
        ge=16,
        le=2048,
        description="Maximum number of new tokens for Qwen to generate"
    )


class AnalyzeResponse(BaseModel):
    answer: str = Field(..., description="Synthesized analysis response from Qwen")


class StatusResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    qwen_configured: bool
    firebase_configured: bool


class RegisterRequest(BaseModel):
    email: str
    password: str = Field(..., min_length=6)


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    email: str

class StatusResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    qwen_configured: bool
    firebase_configured: bool
