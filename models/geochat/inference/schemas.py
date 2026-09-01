from __future__ import annotations

from pydantic import BaseModel, Field


class GeoChatResponse(BaseModel):
    response: str
    model: str
    modality: str


class GeoChatHealthResponse(BaseModel):
    status: str
    model: str


class GeoChatSARResponse(BaseModel):
    response: str
    model: str
    modality: str = "sar"