"""
api/schemas.py

API-facing request/response models. Kept separate from core/models.py:
core/models.py is the pipeline's internal domain layer (also used by
IdentityManager, FusionEngine, Repository), while these are the HTTP
contract -- the two are allowed to diverge (e.g. Paginated[T], the login
request shape) without dragging API concerns into the pipeline's domain
models, per CLAUDE.md Sec 6's domain/schemas separation.
"""
from __future__ import annotations

from datetime import datetime
from typing import Generic, Optional, TypeVar
from uuid import UUID

from pydantic import BaseModel, Field

from core.models import BehaviorSeverity, SessionStatus, UserRole

T = TypeVar("T")


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class Paginated(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    role: str


class StudentOut(BaseModel):
    student_id: UUID
    display_name: str
    external_ref: Optional[str] = None
    enrolled_at: datetime
    active: bool


class EnrollmentResult(BaseModel):
    student: StudentOut
    accepted_images: int
    skipped_images: list[str] = Field(default_factory=list)


class SessionOut(BaseModel):
    session_id: str
    student_id: Optional[UUID] = None
    tracker_id: int
    camera_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: SessionStatus


class EmotionEventOut(BaseModel):
    event_id: int
    session_id: Optional[str] = None
    student_id: Optional[UUID] = None
    timestamp: datetime
    emotions: dict[str, float]
    dominant_emotion: Optional[str] = None


class PoseEventOut(BaseModel):
    event_id: int
    session_id: Optional[str] = None
    student_id: Optional[UUID] = None
    timestamp: datetime
    slumped_score: float
    rigidity_score: float
    fidgeting_score: float


class BehaviorEventOut(BaseModel):
    event_id: int
    session_id: Optional[str] = None
    student_id: Optional[UUID] = None
    timestamp: datetime
    event_type: str
    payload: dict = Field(default_factory=dict)
    severity: BehaviorSeverity


class PendingCandidateOut(BaseModel):
    """A high-confidence resolve() match awaiting human approve/reject
    (see services/identity/manager.py's _maybe_queue_auto_update_candidate).
    image_b64 is included directly (not a separate endpoint) -- the review
    queue is expected to be low-volume (rate-limited to ~1/student/hour),
    so inlining the image is simpler than a dedicated binary route."""

    candidate_id: UUID
    student_id: UUID
    student_display_name: str
    tracker_id: Optional[int] = None
    session_id: Optional[str] = None
    distance: float
    captured_at: datetime
    image_b64: str


class UserOut(BaseModel):
    user_id: UUID
    username: str
    role: UserRole
    active: bool
    created_at: datetime
    assigned_student_count: int = 0


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: UserRole = UserRole.REVIEWER


class AssignmentsRequest(BaseModel):
    student_ids: list[UUID]


class AssignmentsOut(BaseModel):
    student_ids: list[UUID]
