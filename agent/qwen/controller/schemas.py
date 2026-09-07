from __future__ import annotations

from enum import Enum
from typing import Any, Literal
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


class ToolStatus(str, Enum):
    AVAILABLE = "available"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    AUTHENTICATION_ERROR = "authentication_error"
    INVALID_INPUT = "invalid_input"
    UPSTREAM_ERROR = "upstream_error"


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
        value = value.strip()
        parsed = urlparse(value)
        # Backend inputs are signed HTTPS URLs.  Local paths are also valid
        # for direct Gradio testing and for the executor's local test mode.
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return value
        if parsed.scheme == "" and value and not value.startswith("//"):
            return value
        raise ValueError("image references must be valid URLs or local paths")


class SARFiles(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vv: ImageReference
    vh: ImageReference


class RelationshipMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["single", "temporal", "bi_temporal", "cross_modal", "mixed"] | None = None
    spatially_corresponding: bool | None = None
    co_registered: bool | None = None
    relationship: Literal["single", "temporal", "cross_modal", "mixed"] | None = None

    @property
    def kind(self) -> str | None:
        return self.relationship or self.type


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    modality: Literal["optical", "multispectral", "sar"]
    acquisition_time: str | None = None
    image: ImageReference | None = None
    sar: SARFiles | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_payload(self) -> "Observation":
        if self.modality == "sar":
            if self.sar is None or self.image is not None:
                raise ValueError("SAR observations require VV and VH and no single image")
        elif self.image is None or self.sar is not None:
            raise ValueError("non-SAR observations require one image and no SAR channels")
        return self

    @property
    def physical_files(self) -> tuple[ImageReference, ...]:
        if self.sar is not None:
            return self.sar.vv, self.sar.vh
        return (self.image,) if self.image is not None else ()


class InputManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observations: list[Observation] = Field(min_length=1, max_length=4)
    relationship: RelationshipMetadata = Field(default_factory=RelationshipMetadata)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_manifest(self) -> "InputManifest":
        ids = [observation.id for observation in self.observations]
        if len(ids) != len(set(ids)):
            raise ValueError("observation IDs must be unique")
        physical_count = sum(len(observation.physical_files) for observation in self.observations)
        if physical_count > 4:
            raise ValueError("at most four physical image files are supported")
        relation = self.relationship.kind
        modalities = [observation.modality for observation in self.observations]
        if relation in {"temporal", "bi_temporal"}:
            if len(self.observations) != 2:
                raise ValueError("temporal relationships require exactly two observations")
            if not all(observation.acquisition_time for observation in self.observations):
                raise ValueError("temporal relationships require acquisition_time for both observations")
        if relation == "cross_modal":
            if len(self.observations) != 2 or set(modalities) not in ({"optical", "sar"}, {"multispectral", "sar"}):
                raise ValueError("cross_modal relationships require one optical/multispectral and one SAR observation")
        return self

    @property
    def physical_files(self) -> tuple[ImageReference, ...]:
        return tuple(
            image
            for observation in self.observations
            for image in observation.physical_files
        )


def build_input_manifest(
    physical_refs: list[str],
    raw_manifest: dict[str, Any] | InputManifest | None = None,
    *,
    metadata: dict[str, Any] | None = None,
    authorized_refs: dict[str, str | None] | None = None,
) -> InputManifest:
    """Convert transport-level physical references into logical observations.

    This is the only place where physical files are associated with logical
    observations.  A manifest may refer to a direct Gradio file by its
    zero-based ``physical_index`` or to a trusted backend reference ID.
    Filenames are deliberately never inspected.
    """
    refs = [str(value).strip() for value in physical_refs if str(value).strip()]
    if not refs:
        raise ValueError("at least one image input is required")
    if len(refs) > 4:
        raise ValueError("at most four physical image files are supported")

    if isinstance(raw_manifest, InputManifest):
        if len(raw_manifest.physical_files) != len(refs):
            raise ValueError("manifest physical files must match image inputs")
        return raw_manifest

    payload = raw_manifest if isinstance(raw_manifest, dict) else (metadata or {})
    observation_payloads = payload.get("observations")
    if observation_payloads is None:
        # Keep the old explicit channel-role contract usable for trusted
        # callers, but do not infer it from filenames or file extensions.
        if payload.get("sar_channels") == ["vv_t1", "vh_t1", "vv_t2", "vh_t2"] and len(refs) == 4:
            observation_payloads = [
                {"id": "observation_t1", "modality": "sar", "acquisition_time": "t1", "sar": {"vv": {"physical_index": 0}, "vh": {"physical_index": 1}}},
                {"id": "observation_t2", "modality": "sar", "acquisition_time": "t2", "sar": {"vv": {"physical_index": 2}, "vh": {"physical_index": 3}}},
            ]
        elif len(refs) == 1:
            observation_payloads = [{"id": "observation_1", "modality": "optical", "image": {"physical_index": 0}}]
        else:
            # Keep older API callers on a structured incompatibility path.
            # The Gradio adapter rejects this ambiguous shape explicitly.
            observation_payloads = [
                {"id": f"observation_{index}", "modality": "optical", "image": {"physical_index": index - 1}}
                for index in range(1, len(refs) + 1)
            ]

    if not isinstance(observation_payloads, list):
        raise ValueError("manifest observations must be a list")

    id_to_ref = {f"image_{index + 1}": value for index, value in enumerate(refs)}
    if authorized_refs:
        id_to_ref = {key: value for key, value in authorized_refs.items() if value}

    def resolve(value: Any, fallback: str) -> ImageReference:
        item = value if isinstance(value, dict) else {"id": value}
        physical_index = item.get("physical_index")
        if physical_index is not None:
            if not isinstance(physical_index, int) or not 0 <= physical_index < len(refs):
                raise ValueError(f"manifest physical_index for {fallback!r} is out of range")
            if authorized_refs:
                authorized_items = [(key, value) for key, value in authorized_refs.items() if value]
                ref_id, url = authorized_items[physical_index]
            else:
                ref_id, url = f"image_{physical_index + 1}", refs[physical_index]
        else:
            ref_id = str(item.get("id", fallback))
            url = item.get("url") or item.get("path") or id_to_ref.get(ref_id)
            if not url:
                if authorized_refs:
                    raise ValueError(f"Manifest references an unauthorized image ID: {ref_id}")
                raise ValueError(f"manifest reference {ref_id!r} has no authorized physical reference")
            if authorized_refs:
                url = id_to_ref.get(ref_id)
            elif url not in refs:
                raise ValueError(f"manifest reference {ref_id!r} is not authorized")
        return ImageReference(
            image_id=ref_id,
            url=url,
            modality=item.get("modality"),
            timestamp=item.get("timestamp"),
            bands=item.get("bands"),
            spatially_corresponding=item.get("spatially_corresponding"),
        )

    observations: list[Observation] = []
    for index, item in enumerate(observation_payloads, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"manifest observation {index} must be an object")
        observation_id = str(item.get("id", f"observation_{index}"))
        modality = item.get("modality")
        if modality is None:
            if item.get("sar") is not None:
                modality = "sar"
            elif item.get("image") is not None:
                modality = "optical"
            else:
                raise ValueError(f"observation {observation_id!r} must declare modality or an image/sar payload")
        if modality == "sar":
            sar = item.get("sar") or {}
            observations.append(Observation(
                id=observation_id,
                modality="sar",
                acquisition_time=item.get("acquisition_time"),
                sar=SARFiles(
                    vv=resolve(sar.get("vv"), f"{observation_id}_vv"),
                    vh=resolve(sar.get("vh"), f"{observation_id}_vh"),
                ),
                metadata=item.get("metadata", {}),
            ))
        else:
            observations.append(Observation(
                id=observation_id,
                modality=modality,
                acquisition_time=item.get("acquisition_time"),
                image=resolve(item.get("image"), f"{observation_id}_image"),
                metadata=item.get("metadata", {}),
            ))

    relationship = payload.get("relationship") or {}
    return InputManifest(
        observations=observations,
        relationship=RelationshipMetadata.model_validate(relationship),
        metadata=payload.get("metadata") or {},
    )


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_request: str = Field(min_length=1)
    signed_image_urls: list[str] = Field(default_factory=list, max_length=4)
    local_image_paths: list[str] = Field(default_factory=list, max_length=4)
    metadata: dict[str, Any] = Field(default_factory=dict)
    manifest: InputManifest | dict[str, Any] | None = None
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
        cleaned = [value.strip() for value in values]
        for value in cleaned:
            parsed = urlparse(value)
            if parsed.scheme != "https" or not parsed.netloc:
                raise ValueError("signed image URLs must be valid HTTPS URLs")
        return cleaned

    @field_validator("local_image_paths")
    @classmethod
    def validate_local_paths(cls, values: list[str]) -> list[str]:
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("local image paths must not be empty")
        return [value.strip() for value in values]

    @model_validator(mode="after")
    def build_or_validate_manifest(self) -> "AnalysisRequest":
        physical_count = len(self.signed_image_urls) + len(self.local_image_paths)
        if not physical_count:
            raise ValueError("at least one image input is required")
        if self.manifest is None:
            self.manifest = build_input_manifest(self.physical_image_refs, metadata=self.metadata)
        elif isinstance(self.manifest, dict):
            self.manifest = build_input_manifest(self.physical_image_refs, raw_manifest=self.manifest)
        elif len(self.manifest.physical_files) != physical_count:
            raise ValueError("manifest physical files must match image inputs")
        return self

    @property
    def physical_image_refs(self) -> list[str]:
        return [*self.signed_image_urls, *self.local_image_paths]

class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["geochat", "teochat", "m2cd", "pseudo_rgb"]
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
    success: bool = True
    status: ToolStatus = ToolStatus.COMPLETED
    error_summary: str | None = None


class ExecutionTrace(BaseModel):
    step: int
    tool: str | None = None
    operation: str
    status: ToolStatus | str
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
