"""
core/models.py

Pydantic data models shared across the identity subsystem (and, later,
the rest of the pipeline). Keeping these centralized means every service
speaks the same typed contract instead of passing raw dicts around.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, Field, ConfigDict, field_validator


# --------------------------------------------------------------------------- #
# Face quality
# --------------------------------------------------------------------------- #

class QualityRejectReason(str, Enum):
    BLUR = "blur"
    ILLUMINATION_LOW = "illumination_low"
    ILLUMINATION_HIGH = "illumination_high"
    POSE_ANGLE = "pose_angle"
    OCCLUSION = "occlusion"
    TOO_SMALL = "too_small"
    NO_FACE = "no_face"


class FaceQualityReport(BaseModel):
    """Result of pre-embedding quality gating for a single face crop."""

    passed: bool
    blur_score: float = Field(..., description="Variance of Laplacian; higher = sharper")
    brightness: float = Field(..., description="Mean pixel intensity 0-255")
    yaw_deg: Optional[float] = Field(None, description="Estimated head yaw in degrees")
    pitch_deg: Optional[float] = Field(None, description="Estimated head pitch in degrees")
    occlusion_score: float = Field(..., description="0 = fully visible, 1 = fully occluded")
    face_size_px: int = Field(..., description="Shorter side of the detected face box, in pixels")
    reasons: list[QualityRejectReason] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #

class FaceEmbedding(BaseModel):
    """A single ArcFace embedding tied to a student, with provenance."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    embedding_id: UUID = Field(default_factory=uuid4)
    student_id: UUID
    vector: list[float] = Field(..., description="512-d L2-normalized ArcFace embedding")
    quality: FaceQualityReport
    source: str = Field("enrollment", description="'enrollment' or 'auto_update'")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("vector")
    @classmethod
    def _check_dim(cls, v: list[float]) -> list[float]:
        # Deliberately not hardcoded to 512: the model layer stays
        # embedder-agnostic (ArcFace/buffalo_l produces 512-d in
        # production, but tests and future embedders may differ).
        # Dimension *consistency* within one VectorIndex is enforced at
        # the index layer instead.
        if len(v) == 0:
            raise ValueError("Embedding vector must not be empty")
        return v

    def as_array(self) -> np.ndarray:
        return np.asarray(self.vector, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Students / identities
# --------------------------------------------------------------------------- #

class Student(BaseModel):
    student_id: UUID = Field(default_factory=uuid4)
    display_name: str
    external_ref: Optional[str] = Field(None, description="e.g. school roll number")
    enrolled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    active: bool = True


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #

class MatchStatus(str, Enum):
    MATCHED = "matched"
    UNKNOWN = "unknown"
    REJECTED_QUALITY = "rejected_quality"
    CACHE_HIT = "cache_hit"


class IdentityMatchResult(BaseModel):
    status: MatchStatus
    student_id: Optional[UUID] = None
    distance: Optional[float] = Field(None, description="Lower = more similar (cosine distance)")
    confidence: Optional[float] = Field(None, description="1 - normalized distance, 0..1")
    quality: Optional[FaceQualityReport] = None
    tracker_id: Optional[int] = None
    latency_ms: Optional[float] = None


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #

class PoseReading(BaseModel):
    slumped_score: float = 0.0
    rigidity_score: float = 0.0
    fidgeting_score: float = 0.0


class BehaviorSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class BehaviorFlag(BaseModel):
    """A named condition the Fusion Engine detected (e.g. sustained
    negative-affect + slumped posture), destined for behavior_events."""

    event_type: str
    severity: BehaviorSeverity
    detail: str


class FusionEvent(BaseModel):
    """The single unified output of combining identity + emotion + pose for
    one tracked person in one fusion cycle. This -- not raw per-frame
    emotion/pose dicts -- is what gets written to the database, so a
    single frame's noise never becomes a database write on its own."""

    student_id: Optional[UUID] = None
    tracker_id: int
    session_id: Optional[str] = None
    emotions: dict[str, float] = Field(default_factory=dict)
    pose: PoseReading = Field(default_factory=PoseReading)
    identity_confidence: Optional[float] = None
    behavior_flags: list[BehaviorFlag] = Field(default_factory=list)
    sample_count: int = Field(0, description="Number of raw emotion/pose samples averaged into this event")
    window_start: datetime
    window_end: datetime