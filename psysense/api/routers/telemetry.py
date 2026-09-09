"""
api/routers/telemetry.py

Read-only endpoints over sessions/emotion/pose/behavior data -- everything
the dashboard needs to render trends and alerts. All list endpoints are
paginated and support explicit time-range filtering (CLAUDE.md Sec 20);
none return an unbounded table scan.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends

from api.deps import get_config, get_repo, require_admin
from api.pagination import resolve_pagination
from api.schemas import BehaviorEventOut, EmotionEventOut, Paginated, PoseEventOut, SessionOut
from core.config import AppConfig
from db.repository.repository import Repository

router = APIRouter(prefix="/api/v1", tags=["telemetry"], dependencies=[Depends(require_admin)])


@router.get("/sessions", response_model=Paginated[SessionOut])
def list_sessions(
    student_id: Optional[UUID] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    repo: Repository = Depends(get_repo),
    config: AppConfig = Depends(get_config),
) -> Paginated[SessionOut]:
    resolved_limit, resolved_offset = resolve_pagination(limit, offset, config.api)
    items = repo.list_sessions(limit=resolved_limit, offset=resolved_offset, student_id=student_id, since=since, until=until)
    total = repo.count_sessions(student_id=student_id, since=since, until=until)
    return Paginated(
        items=[SessionOut(**s.model_dump()) for s in items], total=total, limit=resolved_limit, offset=resolved_offset,
    )


@router.get("/students/{student_id}/emotions", response_model=Paginated[EmotionEventOut])
def list_student_emotions(
    student_id: UUID,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    repo: Repository = Depends(get_repo),
    config: AppConfig = Depends(get_config),
) -> Paginated[EmotionEventOut]:
    resolved_limit, resolved_offset = resolve_pagination(limit, offset, config.api)
    items = repo.list_emotion_events(limit=resolved_limit, offset=resolved_offset, student_id=student_id, since=since, until=until)
    total = repo.count_emotion_events(student_id=student_id, since=since, until=until)
    return Paginated(
        items=[EmotionEventOut(**e.model_dump()) for e in items], total=total, limit=resolved_limit, offset=resolved_offset,
    )


@router.get("/students/{student_id}/pose", response_model=Paginated[PoseEventOut])
def list_student_pose(
    student_id: UUID,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    repo: Repository = Depends(get_repo),
    config: AppConfig = Depends(get_config),
) -> Paginated[PoseEventOut]:
    resolved_limit, resolved_offset = resolve_pagination(limit, offset, config.api)
    items = repo.list_pose_events(limit=resolved_limit, offset=resolved_offset, student_id=student_id, since=since, until=until)
    total = repo.count_pose_events(student_id=student_id, since=since, until=until)
    return Paginated(
        items=[PoseEventOut(**p.model_dump()) for p in items], total=total, limit=resolved_limit, offset=resolved_offset,
    )


@router.get("/students/{student_id}/behavior-events", response_model=Paginated[BehaviorEventOut])
def list_student_behavior_events(
    student_id: UUID,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    severity: Optional[str] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    repo: Repository = Depends(get_repo),
    config: AppConfig = Depends(get_config),
) -> Paginated[BehaviorEventOut]:
    resolved_limit, resolved_offset = resolve_pagination(limit, offset, config.api)
    items = repo.list_behavior_events(
        limit=resolved_limit, offset=resolved_offset, student_id=student_id, since=since, until=until, severity=severity,
    )
    total = repo.count_behavior_events(student_id=student_id, since=since, until=until, severity=severity)
    return Paginated(
        items=[BehaviorEventOut(**b.model_dump()) for b in items], total=total, limit=resolved_limit, offset=resolved_offset,
    )


@router.get("/behavior-events", response_model=Paginated[BehaviorEventOut])
def list_recent_behavior_events(
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    severity: Optional[str] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    repo: Repository = Depends(get_repo),
    config: AppConfig = Depends(get_config),
) -> Paginated[BehaviorEventOut]:
    """Recent flagged events across ALL students -- the dashboard's alerts
    feed. Deliberately separate from the per-student endpoint above rather
    than making student_id optional there, so the two intents (one
    student's history vs. a cross-student alert stream) stay distinct
    routes with distinct semantics."""
    resolved_limit, resolved_offset = resolve_pagination(limit, offset, config.api)
    items = repo.list_behavior_events(
        limit=resolved_limit, offset=resolved_offset, student_id=None, since=since, until=until, severity=severity,
    )
    total = repo.count_behavior_events(student_id=None, since=since, until=until, severity=severity)
    return Paginated(
        items=[BehaviorEventOut(**b.model_dump()) for b in items], total=total, limit=resolved_limit, offset=resolved_offset,
    )
