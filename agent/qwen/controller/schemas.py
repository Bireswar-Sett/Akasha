from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ============================================================================
# CORE ABSTRACTION
# ============================================================================

"""
Physical files
    ↓
Logical observations
    ↓
Input relationship
    ↓
Semantic task
    ↓
Tool plan
    ↓
Specialist evidence
    ↓
Final response

Important:
- A SAR observation consists of VV + VH physical files.
- Maximum physical files = 4.
- Qwen never invents image IDs, URLs, credentials, or metadata.
- Tool names are closed-set and executor-controlled.
"""


# ============================================================================
# ENUMS
# ============================================================================


class Modality(str, Enum):
    OPTICAL = "optical"
    MULTISPECTRAL = "multispectral"
    SAR = "sar"


class TaskType(str, Enum):
    """
    Semantic user intent.

    Routing should use task + input relationship + modality,
    rather than isolated keywords.
    """

    VISUAL_QA = "visual_question_answering"
    SCENE_DESCRIPTION = "scene_description"
    REGION_GROUNDING = "text_guided_region_grounding"
    INFORMATION_EXTRACTION = "visual_information_extraction"

    CHANGE_ANALYSIS = "change_analysis"
    CHANGE_DESCRIPTION = "change_description"
    CHANGE_QA = "change_question_answering"
    TEMPORAL_SEMANTIC = "bi_temporal_change_question_answering"
    TEMPORAL_REASONING = "temporal_reasoning"

    CROSS_MODAL_ANALYSIS = "cross_modal_analysis"


class InputConfiguration(str, Enum):
    """
    Semantic relationship between logical observations.

    These are NOT physical-file counts.

    OPTICAL_SAR:
        one optical/multispectral observation + one SAR observation

    BI_TEMPORAL:
        two corresponding observations acquired at different times

    DUAL_SAR:
        two SAR observations, each containing VV + VH
    """

    SINGLE_IMAGE = "single_image"
    OPTICAL_SAR = "optical_sar"
    BI_TEMPORAL = "bi_temporal"
    DUAL_SAR = "dual_sar"

    # Compatibility aliases for older callers / persisted manifests.
    CROSS_MODAL_PAIR = "optical_sar"
    BI_TEMPORAL_PAIR = "bi_temporal"


class RelationshipType(str, Enum):
    SINGLE = "single"
    CROSS_MODAL = "cross_modal"
    TEMPORAL = "temporal"
    TEMPORAL_SEQUENCE = "temporal_sequence"
    HETEROGENEOUS_TEMPORAL = "heterogeneous_temporal"


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


class ToolName(str, Enum):
    GEOCHAT = "geochat"
    TEOCHAT = "teochat"
    M2CD = "m2cd"
    PSEUDO_RGB = "pseudo_rgb"


# ============================================================================
# SPATIAL DATA
# ============================================================================


class BoundingBox(BaseModel):
    """
    Normalized bounding box.

    Coordinates are expected to be in [0, 100].
    """

    model_config = ConfigDict(extra="forbid")

    x_left: float
    y_top: float
    x_right: float
    y_bottom: float
    angle: float = 0.0

    @field_validator(
        "x_left",
        "y_top",
        "x_right",
        "y_bottom",
    )
    @classmethod
    def normalized_coordinate(cls, value: float) -> float:
        if not 0 <= value <= 100:
            raise ValueError(
                "bounding-box coordinates must be between 0 and 100"
            )
        return value

    @model_validator(mode="after")
    def valid_geometry(self) -> "BoundingBox":
        if self.x_left > self.x_right:
            raise ValueError(
                "x_left must be less than or equal to x_right"
            )

        if self.y_top > self.y_bottom:
            raise ValueError(
                "y_top must be less than or equal to y_bottom"
            )

        return self

    def compact(self) -> str:
        return (
            f"{{{self.x_left:g}, {self.y_top:g}, "
            f"{self.x_right:g}, {self.y_bottom:g}|{self.angle:g}}}"
        )


class ImageRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str = Field(min_length=1)
    bounding_box: BoundingBox


# ============================================================================
# PHYSICAL IMAGE REFERENCE
# ============================================================================


class ImageReference(BaseModel):
    """
    One physical image resource.

    For SAR, one logical observation contains two references:

        VV
        VH

    URLs may be:
    - HTTPS signed URLs
    - local filesystem paths for direct Space testing
    """

    model_config = ConfigDict(extra="forbid")

    image_id: str = Field(min_length=1)
    url: str = Field(min_length=1)

    role: str | None = None
    filename: str | None = None
    format: str | None = None
    size: int | None = Field(default=None, ge=0)

    modality: Modality | None = None
    timestamp: str | None = None
    bands: list[str] | None = None
    polarization: Literal["VV", "VH"] | None = None

    spatially_corresponding: bool | None = None
    co_registered: bool | None = None

    physical_index: int | None = Field(
        default=None,
        ge=0,
        le=3,
    )

    @field_validator("url")
    @classmethod
    def validate_resource_reference(cls, value: str) -> str:
        value = value.strip()

        parsed = urlparse(value)

        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return value

        # Local filesystem path used during direct Space testing.
        if parsed.scheme == "" and value and not value.startswith("//"):
            return value

        raise ValueError(
            "image reference must be an HTTPS URL or local filesystem path"
        )


# ============================================================================
# SAR OBSERVATION
# ============================================================================


class SARFiles(BaseModel):
    """
    Physical files belonging to ONE logical SAR observation.

    VV and VH are inseparable at the observation level.
    """

    model_config = ConfigDict(extra="forbid")

    vv: ImageReference
    vh: ImageReference

    @model_validator(mode="after")
    def validate_channels(self) -> "SARFiles":
        if self.vv.image_id == self.vh.image_id:
            raise ValueError(
                "VV and VH must reference different physical files"
            )

        if self.vv.polarization not in {None, "VV"}:
            raise ValueError(
                "VV channel must have polarization VV"
            )

        if self.vh.polarization not in {None, "VH"}:
            raise ValueError(
                "VH channel must have polarization VH"
            )

        return self

    @property
    def physical_files(self) -> tuple[ImageReference, ImageReference]:
        return self.vv, self.vh


# ============================================================================
# LOGICAL OBSERVATION
# ============================================================================


class Observation(BaseModel):
    """
    One logical Earth-observation acquisition.

    Optical:
        one physical image

    Multispectral:
        one physical image

    SAR:
        VV + VH physical files
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    modality: Modality
    acquisition_time: str | None = None

    image: ImageReference | None = None
    sar: SARFiles | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_payload(self) -> "Observation":
        if self.modality == Modality.SAR:
            if self.sar is None:
                raise ValueError(
                    "SAR observations require both VV and VH files"
                )

            if self.image is not None:
                raise ValueError(
                    "SAR observations cannot also contain a single image"
                )

        else:
            if self.image is None:
                raise ValueError(
                    "optical/multispectral observations require one image"
                )

            if self.sar is not None:
                raise ValueError(
                    "optical/multispectral observations cannot contain SAR channels"
                )

        return self

    @property
    def physical_files(self) -> tuple[ImageReference, ...]:
        if self.sar is not None:
            return self.sar.physical_files

        if self.image is not None:
            return (self.image,)

        return ()

    @property
    def physical_file_count(self) -> int:
        return len(self.physical_files)


# ============================================================================
# RELATIONSHIP METADATA
# ============================================================================


class RelationshipMetadata(BaseModel):
    """
    Describes how logical observations relate to one another.

    extra="allow" is intentional for forward-compatible backend metadata.
    """

    model_config = ConfigDict(extra="allow")

    type: RelationshipType | None = None
    relationship: RelationshipType | None = None

    spatially_corresponding: bool | None = None
    co_registered: bool | None = None
    same_geographic_area: bool | None = None

    image_t1: str | None = None
    image_t2: str | None = None

    optical_image_id: str | None = None
    sar_image_id: str | None = None

    @property
    def kind(self) -> RelationshipType | None:
        return self.relationship or self.type


# ============================================================================
# INPUT MANIFEST
# ============================================================================


class InputManifest(BaseModel):
    """
    Authoritative semantic representation of supplied imagery.

    physical_files:
        transport/resource view

    observations:
        logical/semantic view

    Maximum physical files = 4.
    """

    model_config = ConfigDict(extra="forbid")

    physical_files: list[ImageReference] = Field(
        default_factory=list,
        max_length=4,
    )

    observations: list[Observation] = Field(
        min_length=1,
        max_length=4,
    )

    relationship: RelationshipMetadata = Field(
        default_factory=RelationshipMetadata
    )

    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_manifest(self) -> "InputManifest":
        observation_ids = [
            observation.id
            for observation in self.observations
        ]

        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError(
                "observation IDs must be unique"
            )

        # Build physical registry when omitted.
        if not self.physical_files:
            files: list[ImageReference] = []
            seen: set[str] = set()

            for observation in self.observations:
                for image in observation.physical_files:
                    if image.image_id not in seen:
                        files.append(image)
                        seen.add(image.image_id)

            self.physical_files = files

        registry_ids = [
            image.image_id
            for image in self.physical_files
        ]

        if len(registry_ids) != len(set(registry_ids)):
            raise ValueError(
                "physical file IDs must be unique"
            )

        if len(self.physical_files) > 4:
            raise ValueError(
                "at most four physical image files are supported"
            )

        referenced_ids = [
            image.image_id
            for observation in self.observations
            for image in observation.physical_files
        ]

        if len(referenced_ids) != len(set(referenced_ids)):
            raise ValueError(
                "a physical file cannot be assigned to more than one observation"
            )

        if set(referenced_ids) != set(registry_ids):
            raise ValueError(
                "physical file registry must exactly match observation references"
            )

        relation = self.relationship.kind

        modalities = [
            observation.modality
            for observation in self.observations
        ]

        if relation == RelationshipType.SINGLE:
            if len(self.observations) != 1:
                raise ValueError(
                    "single relationship requires exactly one observation"
                )

        elif relation == RelationshipType.CROSS_MODAL:
            if len(self.observations) != 2:
                raise ValueError(
                    "cross-modal relationship requires exactly two observations"
                )

            modality_set = set(modalities)

            if modality_set not in {
                {Modality.OPTICAL, Modality.SAR},
                {Modality.MULTISPECTRAL, Modality.SAR},
            }:
                raise ValueError(
                    "cross-modal relationship requires one optical/multispectral "
                    "and one SAR observation"
                )

        elif relation == RelationshipType.TEMPORAL:
            if len(self.observations) != 2:
                raise ValueError(
                    "temporal relationships require exactly two observations"
                )

            if not all(
                observation.acquisition_time
                for observation in self.observations
            ):
                raise ValueError(
                    "temporal relationships require acquisition_time "
                    "for every observation"
                )

        elif relation == RelationshipType.TEMPORAL_SEQUENCE:
            if len(self.observations) < 2:
                raise ValueError(
                    "temporal sequences require at least two observations"
                )

            if not all(
                observation.acquisition_time
                for observation in self.observations
            ):
                raise ValueError(
                    "temporal sequences require acquisition_time "
                    "for every observation"
                )

        elif relation == RelationshipType.HETEROGENEOUS_TEMPORAL:
            if len(self.observations) != 2:
                raise ValueError(
                    "heterogeneous temporal relationships require exactly two observations"
                )

            if not all(
                observation.acquisition_time
                for observation in self.observations
            ):
                raise ValueError(
                    "heterogeneous temporal relationships require "
                    "acquisition_time for both observations"
                )

        return self


# ============================================================================
# INPUT MANIFEST BUILDER
# ============================================================================


def build_input_manifest(
    physical_refs: list[str],
    raw_manifest: dict[str, Any] | InputManifest | None = None,
    *,
    metadata: dict[str, Any] | None = None,
    authorized_refs: dict[str, str | None] | None = None,
) -> InputManifest:
    """
    Convert physical transport references into logical observations.

    This function deliberately does NOT infer modality from filenames.

    For multiple physical files, explicit backend-generated logical
    observation metadata is required.

    authorized_refs:
        maps trusted resource IDs to authorized URLs.
    """

    refs = [
        str(value).strip()
        for value in physical_refs
        if str(value).strip()
    ]

    if not refs:
        raise ValueError("at least one image input is required")

    if len(refs) > 4:
        raise ValueError(
            "at most four physical image files are supported"
        )

    if isinstance(raw_manifest, InputManifest):
        if len(raw_manifest.physical_files) != len(refs):
            raise ValueError(
                "manifest physical file count must match input file count"
            )
        return raw_manifest

    if isinstance(raw_manifest, dict):
        payload: dict[str, Any] = dict(raw_manifest)
    elif metadata:
        payload = dict(metadata)
    else:
        payload = {}

    observation_payloads = payload.get("observations")

    # ------------------------------------------------------------------
    # Legacy optical-only payload.
    # ------------------------------------------------------------------

    if observation_payloads is None:
        legacy_images = payload.get("images")

        if isinstance(legacy_images, list):
            if all(
                isinstance(item, dict)
                and item.get("modality") in {
                    "optical",
                    "multispectral",
                }
                for item in legacy_images
            ):
                observation_payloads = []

                for index, item in enumerate(legacy_images):
                    image_id = str(
                        item.get("id")
                        or f"file_{index}"
                    )

                    observation_payloads.append(
                        {
                            "id": str(
                                item.get("observation_id")
                                or f"observation_{index + 1}"
                            ),
                            "modality": item.get("modality"),
                            "acquisition_time": item.get(
                                "acquisition_time"
                                or item.get("timestamp")
                            ),
                            "image": {
                                "id": image_id,
                                "physical_index": index,
                                "filename": item.get("filename"),
                                "format": item.get("format"),
                                "size": item.get("size"),
                                "timestamp": item.get("timestamp"),
                                "bands": item.get("bands"),
                            },
                        }
                    )

    # ------------------------------------------------------------------
    # Single-image fallback.
    #
    # A single physical image may default to optical.
    # Multiple physical files require explicit grouping.
    # ------------------------------------------------------------------

    if observation_payloads is None:
        if len(refs) == 1:
            observation_payloads = [
                {
                    "id": "observation_1",
                    "modality": "optical",
                    "image": {
                        "physical_index": 0,
                    },
                }
            ]
        else:
            raise ValueError(
                "A backend-generated logical observation manifest "
                "is required when multiple physical files are supplied"
            )

    if not isinstance(observation_payloads, list):
        raise ValueError(
            "manifest observations must be a list"
        )

    # ------------------------------------------------------------------
    # Physical registry.
    # ------------------------------------------------------------------

    registry = payload.get("physical_files")

    if registry is not None:
        if (
            not isinstance(registry, list)
            or len(registry) != len(refs)
        ):
            raise ValueError(
                "manifest physical_files must match physical input count"
            )
    else:
        registry = [
            {
                "id": f"file_{index}",
                "physical_index": index,
            }
            for index in range(len(refs))
        ]

    # ------------------------------------------------------------------
    # Map trusted resource IDs to authorized URLs.
    # ------------------------------------------------------------------

    id_to_ref: dict[str, str] = {}

    for index, descriptor in enumerate(registry):
        if not isinstance(descriptor, dict):
            raise ValueError(
                "manifest physical_files entries must be objects"
            )

        ref_id = str(
            descriptor.get("id")
            or f"file_{index}"
        )

        id_to_ref[ref_id] = refs[index]

    if authorized_refs:
        authorized_clean = {
            key: value
            for key, value in authorized_refs.items()
            if value
        }

        if len(authorized_clean) != len(refs):
            raise ValueError(
                "authorized physical references must match "
                "the supplied physical input count"
            )

        id_to_ref = {
            str(key): str(value)
            for key, value in authorized_clean.items()
        }

    # ------------------------------------------------------------------
    # Resolve one physical reference.
    # ------------------------------------------------------------------

    def resolve(
        value: Any,
        fallback_id: str,
    ) -> ImageReference:
        item = (
            dict(value)
            if isinstance(value, dict)
            else {"id": value}
        )

        physical_index = item.get("physical_index")

        if physical_index is not None:
            if not isinstance(physical_index, int):
                raise ValueError(
                    f"physical_index for {fallback_id!r} must be an integer"
                )

            if not 0 <= physical_index < len(refs):
                raise ValueError(
                    f"physical_index for {fallback_id!r} is out of range"
                )

            registry_descriptor = registry[physical_index]

            ref_id = str(
                registry_descriptor.get("id")
                or f"file_{physical_index}"
            )

            if authorized_refs:
                if ref_id not in id_to_ref:
                    raise ValueError(
                        f"physical resource {ref_id!r} is not authorized"
                    )

                url = id_to_ref[ref_id]
            else:
                url = refs[physical_index]

        else:
            ref_id = str(
                item.get("id")
                or item.get("file_id")
                or fallback_id
            )

            # Legacy aliases:
            # image_1 -> physical index 0
            # image_2 -> physical index 1
            if (
                ref_id.startswith("image_")
                and ref_id[6:].isdigit()
                and not authorized_refs
            ):
                alias_index = int(ref_id[6:]) - 1

                if 0 <= alias_index < len(refs):
                    registry_descriptor = registry[alias_index]

                    ref_id = str(
                        registry_descriptor.get("id")
                        or f"file_{alias_index}"
                    )

            url = (
                item.get("url")
                or item.get("path")
                or id_to_ref.get(ref_id)
            )

            if not url:
                raise ValueError(
                    f"manifest reference {ref_id!r} has no authorized physical reference"
                )

            if authorized_refs:
                if ref_id not in id_to_ref:
                    raise ValueError(
                        f"manifest reference {ref_id!r} is not authorized"
                    )

                url = id_to_ref[ref_id]

            elif url not in refs:
                raise ValueError(
                    f"manifest reference {ref_id!r} is not authorized"
                )

        polarization = item.get("polarization")

        if polarization is not None:
            polarization = str(polarization).upper()

        return ImageReference(
            image_id=ref_id,
            url=str(url),
            role=item.get("role"),
            filename=item.get("filename"),
            format=item.get("format"),
            size=item.get("size"),
            modality=item.get("modality"),
            timestamp=item.get("timestamp"),
            bands=item.get("bands"),
            polarization=polarization,
            spatially_corresponding=item.get(
                "spatially_corresponding"
            ),
            co_registered=item.get(
                "co_registered"
            ),
            physical_index=physical_index,
        )

    # ------------------------------------------------------------------
    # Build logical observations.
    # ------------------------------------------------------------------

    observations: list[Observation] = []

    for index, item in enumerate(
        observation_payloads,
        start=1,
    ):
        if not isinstance(item, dict):
            raise ValueError(
                f"manifest observation {index} must be an object"
            )

        observation_id = str(
            item.get("id")
            or f"observation_{index}"
        )

        modality_value = item.get("modality")

        if modality_value is None:
            if item.get("sar") is not None:
                modality_value = "sar"
            elif item.get("image") is not None:
                modality_value = "optical"
            else:
                raise ValueError(
                    f"observation {observation_id!r} must declare modality"
                )

        modality = Modality(modality_value)

        acquisition_time = item.get(
            "acquisition_time"
        )

        if modality == Modality.SAR:
            sar_payload = item.get("sar")

            if not isinstance(sar_payload, dict):
                raise ValueError(
                    f"SAR observation {observation_id!r} "
                    "requires sar.vv and sar.vh"
                )

            observations.append(
                Observation(
                    id=observation_id,
                    modality=modality,
                    acquisition_time=acquisition_time,
                    sar=SARFiles(
                        vv=resolve(
                            sar_payload.get("vv"),
                            f"{observation_id}_vv",
                        ),
                        vh=resolve(
                            sar_payload.get("vh"),
                            f"{observation_id}_vh",
                        ),
                    ),
                    metadata=item.get(
                        "metadata",
                        {},
                    ),
                )
            )

        else:
            observations.append(
                Observation(
                    id=observation_id,
                    modality=modality,
                    acquisition_time=acquisition_time,
                    image=resolve(
                        item.get("image"),
                        f"{observation_id}_image",
                    ),
                    metadata=item.get(
                        "metadata",
                        {},
                    ),
                )
            )

    relationship_payload = (
        payload.get("relationship")
        or {}
    )

    return InputManifest(
        physical_files=[
            ImageReference(
                image_id=str(
                    descriptor.get("id")
                    or f"file_{index}"
                ),
                url=refs[index],
                role=descriptor.get("role"),
                filename=descriptor.get("filename"),
                format=descriptor.get("format"),
                size=descriptor.get("size"),
                modality=descriptor.get("modality"),
                timestamp=descriptor.get("timestamp"),
                bands=descriptor.get("bands"),
                polarization=descriptor.get("polarization"),
                physical_index=index,
            )
            for index, descriptor in enumerate(registry)
        ],
        observations=observations,
        relationship=RelationshipMetadata.model_validate(
            relationship_payload
        ),
        metadata=payload.get("metadata") or {},
    )


# ============================================================================
# ANALYSIS REQUEST
# ============================================================================


class AnalysisRequest(BaseModel):
    """
    Main request received by the Qwen controller.

    Canonical fields:

        user_request
        signed_image_urls
        local_image_paths
        manifest

    Compatibility inputs accepted at the controller boundary:

        user_message
        image_url
        image_urls

    The compatibility inputs are normalized immediately and are not
    allowed to create a second parallel request representation.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    user_request: str = Field(min_length=1)

    signed_image_urls: list[str] = Field(
        default_factory=list,
        max_length=4,
    )

    local_image_paths: list[str] = Field(
        default_factory=list,
        max_length=4,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    manifest: InputManifest | dict[str, Any] | None = None

    user_query: str | None = None

    bounding_boxes: list[BoundingBox] = Field(
        default_factory=list
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        payload = dict(data)

        # Backend/Qwen-service compatibility.
        if not payload.get("user_request"):
            legacy_message = payload.get("user_message")
            if legacy_message:
                payload["user_request"] = legacy_message

        urls = payload.get("signed_image_urls")

        if not urls:
            image_urls = payload.get("image_urls")

            if isinstance(image_urls, list):
                payload["signed_image_urls"] = image_urls
            elif payload.get("image_url"):
                payload["signed_image_urls"] = [
                    payload["image_url"]
                ]

        return payload

    @field_validator("user_request")
    @classmethod
    def non_blank_request(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "user_request must not be empty"
            )

        return value

    @field_validator("signed_image_urls")
    @classmethod
    def validate_signed_urls(
        cls,
        values: list[str],
    ) -> list[str]:
        if len(values) > 4:
            raise ValueError(
                "at most four signed image URLs are supported"
            )

        cleaned: list[str] = []

        for value in values:
            value = value.strip()

            if not value:
                raise ValueError(
                    "signed image URLs must not be empty"
                )

            parsed = urlparse(value)

            if (
                parsed.scheme != "https"
                or not parsed.netloc
            ):
                raise ValueError(
                    "signed image URLs must be valid HTTPS URLs"
                )

            cleaned.append(value)

        return cleaned

    @field_validator("local_image_paths")
    @classmethod
    def validate_local_paths(
        cls,
        values: list[str],
    ) -> list[str]:
        if len(values) > 4:
            raise ValueError(
                "at most four local image paths are supported"
            )

        cleaned = [
            value.strip()
            for value in values
        ]

        if any(not value for value in cleaned):
            raise ValueError(
                "local image paths must not be empty"
            )

        return cleaned

    @model_validator(mode="after")
    def build_or_validate_manifest(
        self,
    ) -> "AnalysisRequest":
        physical_count = (
            len(self.signed_image_urls)
            + len(self.local_image_paths)
        )

        if physical_count == 0:
            raise ValueError(
                "at least one image input is required"
            )

        if physical_count > 4:
            raise ValueError(
                "at most four physical image files are supported"
            )

        if self.manifest is None:
            self.manifest = build_input_manifest(
                self.physical_image_refs,
                metadata=self.metadata,
            )

        elif isinstance(self.manifest, dict):
            self.manifest = build_input_manifest(
                self.physical_image_refs,
                raw_manifest=self.manifest,
            )

        else:
            if (
                len(self.manifest.physical_files)
                != physical_count
            ):
                raise ValueError(
                    "manifest physical file count must match image inputs"
                )

        return self

    @property
    def physical_image_refs(self) -> list[str]:
        return [
            *self.signed_image_urls,
            *self.local_image_paths,
        ]


# ============================================================================
# TOOL PLAN
# ============================================================================


class ToolCall(BaseModel):
    """
    One executable specialist operation.

    Qwen proposes it.
    The Python executor validates and executes it.
    """

    model_config = ConfigDict(extra="forbid")

    name: ToolName

    purpose: str = Field(
        min_length=1
    )

    operation: str = Field(
        min_length=1
    )

    arguments: dict[str, Any] = Field(
        default_factory=dict
    )

    depends_on: list[int] = Field(
        default_factory=list
    )

    bounding_boxes: list[BoundingBox] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_dependencies(self) -> "ToolCall":
        if any(
            dependency < 0
            for dependency in self.depends_on
        ):
            raise ValueError(
                "tool dependency indexes must be non-negative"
            )

        return self


class ToolPlan(BaseModel):
    """
    Complete executable plan.

    Qwen creates the plan.
    The executor remains authoritative over actual execution.
    """

    model_config = ConfigDict(extra="forbid")

    task_type: TaskType

    task_description: str = Field(
        min_length=1
    )

    input_configuration: InputConfiguration

    images_used: list[str] = Field(
        default_factory=list
    )

    calls: list[ToolCall] = Field(
        default_factory=list
    )

    bounding_boxes: list[BoundingBox] = Field(
        default_factory=list
    )

    compatibility_issue: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_tool_dependencies(
        self,
    ) -> "ToolPlan":
        call_count = len(self.calls)

        for index, call in enumerate(self.calls):
            for dependency in call.depends_on:
                if dependency >= call_count:
                    raise ValueError(
                        f"tool call {index} depends on "
                        f"nonexistent call {dependency}"
                    )

                if dependency == index:
                    raise ValueError(
                        f"tool call {index} cannot depend on itself"
                    )

        return self


# ============================================================================
# SPECIALIST EVIDENCE
# ============================================================================


class SpatialEvidence(BaseModel):
    model_config = ConfigDict(extra="allow")

    kind: str
    value: Any


class SpecialistEvidence(BaseModel):
    """
    Normalized specialist result.

    Failed/unavailable tools remain represented so the controller can
    explain what was and was not completed.
    """

    model_config = ConfigDict(extra="allow")

    tool: ToolName

    operation: str

    observation: str | None = None

    interpretation: str | None = None

    uncertainty: str | None = None

    confidence: Any | None = None

    spatial_outputs: list[SpatialEvidence] = Field(
        default_factory=list
    )

    bounding_boxes: list[BoundingBox] = Field(
        default_factory=list
    )

    result: Any = None

    success: bool = True

    status: ToolStatus = ToolStatus.COMPLETED

    error_summary: str | None = None


# ============================================================================
# EXECUTION TRACE
# ============================================================================


class ExecutionTrace(BaseModel):
    """
    Observable audit trail.

    This represents WHAT happened, not hidden chain-of-thought.
    """

    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=1)

    tool: ToolName | None = None

    operation: str

    status: ToolStatus | str

    duration_ms: int | None = Field(
        default=None,
        ge=0,
    )

    details: dict[str, Any] = Field(
        default_factory=dict
    )

    bounding_boxes: list[BoundingBox] = Field(
        default_factory=list
    )


# ============================================================================
# FINAL RESPONSE
# ============================================================================


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str

    task: dict[str, Any] | None = None

    input: dict[str, Any] | None = None

    execution: list[ExecutionTrace] = Field(
        default_factory=list
    )

    evidence: list[SpecialistEvidence] = Field(
        default_factory=list
    )

    answer: str | None = None

    uncertainty: str | None = None

    error: dict[str, Any] | None = None


# ============================================================================
# REQUEST CONTEXT
# ============================================================================


class RequestContext(BaseModel):
    """
    Internal orchestration state.

    Keeps generated artifacts and specialist evidence separate from the
    original request.
    """

    request: AnalysisRequest

    images: list[ImageReference] = Field(
        default_factory=list
    )

    artifacts: dict[str, Any] = Field(
        default_factory=dict
    )

    evidence: list[SpecialistEvidence] = Field(
        default_factory=list
    )