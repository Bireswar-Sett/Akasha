from __future__ import annotations

from enum import Enum
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x_left: float
    y_top: float
    x_right: float
    y_bottom: float
    angle: float

    @field_validator("x_left", "y_top", "x_right", "y_bottom")
    @classmethod
    def normalized_coordinate(cls, value: float) -> float:
        if not 0 <= value <= 100:
            raise ValueError("bounding-box coordinates must be between 0 and 100")
        return value

    @model_validator(mode="after")
    def valid_geometry(self) -> "BoundingBox":
        if self.x_left > self.x_right:
            raise ValueError("x_left must be less than or equal to x_right")
        if self.y_top > self.y_bottom:
            raise ValueError("y_top must be less than or equal to y_bottom")
        return self

    def compact(self) -> str:
        return f"{{{self.x_left:g}, {self.y_top:g}, {self.x_right:g}, {self.y_bottom:g}|{self.angle:g}}}"


class ImageRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str
    bounding_box: BoundingBox


class TaskType(str, Enum):
    VISUAL_QA = "visual_question_answering"
    SCENE_DESCRIPTION = "scene_description"
    REGION_GROUNDING = "text_guided_region_grounding"
    INFORMATION_EXTRACTION = "visual_information_extraction"
    CHANGE_ANALYSIS = "change_analysis"
    TEMPORAL_SEMANTIC = "bi_temporal_change_question_answering"


class InputConfiguration(str, Enum):
    SINGLE_IMAGE = "single_image"
    OPTICAL_SAR = "optical_sar"
    BI_TEMPORAL = "bi_temporal_pair"
    DUAL_SAR = "dual_sar"
    SAR_CHANNELS = "four_url_sar_channels"


class ImageReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str
    url: str
    modality: str | None = None
    timestamp: str | None = None
    bands: list[str] | None = None
    spatially_corresponding: bool | None = None

    @field_validator("url")
    @classmethod
    def https_only(cls, value: str) -> str:
        parsed = urlparse(value.strip())
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("image URLs must be valid HTTPS URLs")
        return value.strip()


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_request: str = Field(min_length=1)
    signed_image_urls: list[str] = Field(min_length=1, max_length=4)
    metadata: dict[str, Any] = Field(default_factory=dict)
    user_query: str | None = None
    bounding_boxes: list[BoundingBox] = Field(default_factory=list)

    @field_validator("user_request")
    @classmethod
    def non_blank_request(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("user_request must not be empty")
        return value.strip()

    @model_validator(mode="after")
    def parse_backend_bbox(self) -> "AnalysisRequest":
        from qwen.controller.bbox import parse_query

        parsed = parse_query(self.user_request)
        self.user_query = parsed.user_query
        self.bounding_boxes = parsed.bounding_boxes
        return self

    @field_validator("signed_image_urls")
    @classmethod
    def validate_urls(cls, values: list[str]) -> list[str]:
        if len(values) > 4:
            raise ValueError("at most four signed image URLs are supported")
        references = [ImageReference(image_id=str(i), url=value) for i, value in enumerate(values)]
        return [reference.url for reference in references]


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    purpose: str
    operation: str
    arguments: dict[str, Any]
    depends_on: list[int] = Field(default_factory=list)
    bounding_boxes: list[BoundingBox] = Field(default_factory=list)


class ToolPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: TaskType
    task_description: str
    input_configuration: InputConfiguration
    images_used: list[str]
    calls: list[ToolCall]
    bounding_boxes: list[BoundingBox] = Field(default_factory=list)
    compatibility_issue: dict[str, Any] | None = None


class SpatialEvidence(BaseModel):
    model_config = ConfigDict(extra="allow")

    kind: str
    value: Any


class SpecialistEvidence(BaseModel):
    model_config = ConfigDict(extra="allow")

    tool: str
    operation: str
    observation: str | None = None
    interpretation: str | None = None
    uncertainty: str | None = None
    confidence: Any | None = None
    spatial_outputs: list[SpatialEvidence] = Field(default_factory=list)
    bounding_boxes: list[BoundingBox] = Field(default_factory=list)
    result: Any = None


class ExecutionTrace(BaseModel):
    step: int
    tool: str | None = None
    operation: str
    status: str
    duration_ms: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    bounding_boxes: list[BoundingBox] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    status: str
    task: dict[str, Any] | None = None
    input: dict[str, Any] | None = None
    execution: list[ExecutionTrace] = Field(default_factory=list)
    evidence: list[SpecialistEvidence] = Field(default_factory=list)
    answer: str | None = None
    uncertainty: str | None = None
    error: dict[str, Any] | None = None


class RequestContext(BaseModel):
    request: AnalysisRequest
    images: list[ImageReference]
    artifacts: dict[str, Any] = Field(default_factory=dict)
    evidence: list[SpecialistEvidence] = Field(default_factory=list)
