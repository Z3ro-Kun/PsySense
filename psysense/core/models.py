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


def _as_utc(v):
    """Timestamps are persisted as naive UTC strings (db/writer.py's
    now_iso()); treat any naive value coming out of the database as UTC
    rather than leaving it ambiguous, so API responses carry explicit
    timezone-aware datetimes."""
    if isinstance(v, str):
        v = datetime.fromisoformat(v)
    if isinstance(v, datetime) and v.tzinfo is None:
        v = v.replace(tzinfo=timezone.utc)
    return v


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
# Sub-admin accounts (reviewers)
# --------------------------------------------------------------------------- #

class UserRole(str, Enum):
    ADMIN = "admin"
    REVIEWER = "reviewer"


class User(BaseModel):
    """A sub-admin account (see db/schema.sql's users table comment for
    why the root env-var admin is deliberately NOT a row here). Never
    carries password_hash -- that's read separately by Repository methods
    that need it for auth verification, so a User accidentally handed to
    an API response can never leak a hash."""

    user_id: UUID = Field(default_factory=uuid4)
    username: str
    role: UserRole = UserRole.REVIEWER
    active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


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


class SessionStatus(str, Enum):
    ACTIVE = "active"
    CLOSED = "closed"
    LOST = "lost"


class TrackingSession(BaseModel):
    """Read model for a row in tracking_sessions -- one tracker-ID's
    lifetime, from first detection to drop/close. Not to be confused with
    FusionEvent, which is a windowed slice of behavior *within* a session."""

    session_id: str
    student_id: Optional[UUID] = None
    tracker_id: int
    camera_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: SessionStatus

    _tz_start = field_validator("start_time", mode="before")(_as_utc)
    _tz_end = field_validator("end_time", mode="before")(_as_utc)


class EmotionEvent(BaseModel):
    """Read model for a row in emotion_events -- one FusionEngine-averaged
    window, not a raw per-frame DeepFace reading."""

    event_id: int
    session_id: Optional[str] = None
    student_id: Optional[UUID] = None
    timestamp: datetime
    emotions: dict[str, float]
    dominant_emotion: Optional[str] = None

    _tz_ts = field_validator("timestamp", mode="before")(_as_utc)


class PoseEventRecord(BaseModel):
    """Read model for a row in pose_events."""

    event_id: int
    session_id: Optional[str] = None
    student_id: Optional[UUID] = None
    timestamp: datetime
    slumped_score: float
    rigidity_score: float
    fidgeting_score: float

    _tz_ts = field_validator("timestamp", mode="before")(_as_utc)


class PendingIdentityCandidate(BaseModel):
    """Read model for a row in pending_identity_candidates -- a
    high-confidence resolve() match awaiting human approve/reject before
    it's added to anyone's profile (see services/identity/manager.py's
    _maybe_queue_auto_update_candidate). Deliberately excludes the raw
    image bytes -- those are fetched separately (Repository.
    get_pending_candidate_image) only by the one API route that actually
    needs to display them, not carried around on every read of this model."""

    candidate_id: UUID
    student_id: UUID
    tracker_id: Optional[int] = None
    session_id: Optional[str] = None
    distance: float
    quality: Optional[FaceQualityReport] = None
    captured_at: datetime

    _tz_ts = field_validator("captured_at", mode="before")(_as_utc)


class BehaviorEventRecord(BaseModel):
    """Read model for a row in behavior_events -- a named condition the
    Fusion Engine flagged (see services/fusion/engine.py), e.g.
    'prolonged_slump'. This is an observed/inferred *signal*, not a
    diagnosis -- see CLAUDE.md Sec 15/39."""

    event_id: int
    session_id: Optional[str] = None
    student_id: Optional[UUID] = None
    timestamp: datetime
    event_type: str
    payload: dict = Field(default_factory=dict)
    severity: BehaviorSeverity

    _tz_ts = field_validator("timestamp", mode="before")(_as_utc)


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