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

    spatially_corresponding: bool | None = None
    co_registered: bool | None = None
    relationship: Literal["single", "temporal", "cross_modal", "mixed"] | None = None


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
        return self

    @property
    def physical_files(self) -> tuple[ImageReference, ...]:
        return tuple(
            image
            for observation in self.observations
            for image in observation.physical_files
        )


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_request: str = Field(min_length=1)
    signed_image_urls: list[str] = Field(default_factory=list, max_length=4)
    local_image_paths: list[str] = Field(default_factory=list, max_length=4)
    metadata: dict[str, Any] = Field(default_factory=dict)
    manifest: InputManifest | None = None
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
            self.manifest = self._legacy_manifest()
        elif len(self.manifest.physical_files) != physical_count:
            raise ValueError("manifest physical files must match image inputs")
        return self

    @property
    def physical_image_refs(self) -> list[str]:
        return [*self.signed_image_urls, *self.local_image_paths]

    def _legacy_manifest(self) -> InputManifest:
        physical_refs = self.physical_image_refs
        channel_roles = self.metadata.get("sar_channels")
        if channel_roles == ["vv_t1", "vh_t1", "vv_t2", "vh_t2"] and len(physical_refs) == 4:
            refs = [ImageReference(image_id=f"image_{index}", url=url, modality="sar") for index, url in enumerate(physical_refs, start=1)]
            return InputManifest(
                observations=[
                    Observation(id="observation_t1", modality="sar", acquisition_time="t1", sar=SARFiles(vv=refs[0], vh=refs[1])),
                    Observation(id="observation_t2", modality="sar", acquisition_time="t2", sar=SARFiles(vv=refs[2], vh=refs[3])),
                ],
                relationship=RelationshipMetadata(relationship="temporal", spatially_corresponding=True),
                metadata={**self.metadata, "_legacy_manifest": False},
            )
        explicit_observations = self.metadata.get("observations")
        if isinstance(explicit_observations, list):
            # The relationship between physical files is supplied by the
            # trusted manifest.  Only URLs are injected here; Qwen never
            # derives pairing from names.
            by_id = {
                f"image_{index}": url
                for index, url in enumerate(physical_refs, start=1)
            }
            def resolve(value: Any, fallback: str) -> ImageReference:
                item = value if isinstance(value, dict) else {"id": value}
                ref_id = str(item.get("id", fallback))
                url = item.get("url") or item.get("path") or by_id.get(ref_id)
                if not url:
                    raise ValueError(f"manifest reference {ref_id!r} has no signed URL")
                if url not in physical_refs:
                    raise ValueError(f"manifest reference {ref_id!r} is not authorized")
                return ImageReference(
                    image_id=ref_id,
                    url=url,
                    modality=item.get("modality"),
                    timestamp=item.get("timestamp"),
                    bands=item.get("bands"),
                )

            observations: list[Observation] = []
            for index, item in enumerate(explicit_observations, start=1):
                modality = item.get("modality")
                observation_id = str(item.get("id", f"observation_{index}"))
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
            return InputManifest(
                observations=observations,
                relationship=RelationshipMetadata.model_validate(self.metadata.get("relationship") or {}),
                metadata={**self.metadata, "_legacy_manifest": False},
            )

        metadata_images = self.metadata.get("images", [])
        metadata_by_id = {
            str(item.get("id", index + 1)): item
            for index, item in enumerate(metadata_images)
            if isinstance(item, dict)
        }
        observations: list[Observation] = []
        for index, url in enumerate(physical_refs, start=1):
            item = metadata_by_id.get(f"image_{index}", metadata_by_id.get(str(index), {}))
            modality = item.get("modality", "optical")
            if modality == "sar" and item.get("vv_url") and item.get("vh_url"):
                observations.append(Observation(
                    id=f"observation_{index}",
                    modality="sar",
                    acquisition_time=item.get("acquisition_time"),
                    sar=SARFiles(
                        vv=ImageReference(image_id=f"image_{index}_vv", url=item["vv_url"]),
                        vh=ImageReference(image_id=f"image_{index}_vh", url=item["vh_url"]),
                    ),
                    metadata=item,
                ))
                continue
            if modality == "sar":
                # Compatibility for the original four-URL Gradio contract.
                # New manifests must provide distinct VV and VH references.
                legacy_ref = ImageReference(image_id=f"image_{index}", url=url, modality="sar")
                observations.append(Observation(
                    id=f"observation_{index}",
                    modality="sar",
                    acquisition_time=item.get("acquisition_time"),
                    sar=SARFiles(vv=legacy_ref, vh=legacy_ref),
                    metadata=item,
                ))
                continue
            observations.append(Observation(
                id=f"observation_{index}",
                modality=modality,
                acquisition_time=item.get("acquisition_time"),
                image=ImageReference(
                    image_id=f"image_{index}",
                    url=url,
                    modality=modality,
                    timestamp=item.get("acquisition_time"),
                    bands=item.get("bands"),
                ),
                metadata=item,
            ))
        pair = self.metadata.get("pair_metadata") or {}
        relationship = self.metadata.get("input_configuration")
        if relationship == "bi_temporal":
            relationship = "temporal"
        elif relationship == "optical_sar":
            relationship = "cross_modal"
        elif relationship not in {"single", "temporal", "cross_modal", "mixed"}:
            relationship = None
        return InputManifest(
            observations=observations,
            relationship=RelationshipMetadata(
                spatially_corresponding=pair.get("spatially_corresponding"),
                co_registered=pair.get("co_registered"),
                relationship=relationship,
            ),
            metadata={**self.metadata, "_legacy_manifest": True},
        )


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
