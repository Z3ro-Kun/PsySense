"""
api/routers/identity_candidates.py

Human-in-the-loop review queue for identity auto-update candidates (see
services/identity/manager.py's _maybe_queue_auto_update_candidate). A
candidate here is a face crop + embedding the pipeline was already very
confident about, but which was deliberately NOT added to anyone's profile
without a human actually looking at the photo and confirming it.

Both admins and reviewers can reach these routes (require_reviewer_or_admin),
but a reviewer only ever sees/approves/rejects candidates for students
they're explicitly assigned to (reviewer_assignments) -- there is no
"reviewer sees everyone" mode. An admin sees and can act on every candidate.
"""
from __future__ import annotations

import base64
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from api.auth import Principal
from api.deps import get_config, get_repo, require_reviewer_or_admin
from api.pagination import resolve_pagination
from api.schemas import Paginated, PendingCandidateOut
from core.config import AppConfig
from core.models import FaceEmbedding
from db.repository.repository import Repository
from logging_.logger import get_logger

logger = get_logger("api.identity_candidates")

router = APIRouter(prefix="/api/v1/identity-candidates", tags=["identity-candidates"])


def _assert_reviewer_can_act(repo: Repository, principal: Principal, student_id: UUID) -> None:
    """Admins can act on any candidate. A reviewer can only act on a
    candidate belonging to a student they're explicitly assigned to --
    checked here, not just in what list_candidates shows, so a reviewer
    can't approve/reject an unassigned student's candidate by guessing
    its candidate_id."""
    if principal.role == "admin":
        return
    if principal.user_id is None or not repo.is_student_assigned_to_reviewer(principal.user_id, student_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not assigned to this student")


@router.get("", response_model=Paginated[PendingCandidateOut])
def list_candidates(
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    repo: Repository = Depends(get_repo),
    config: AppConfig = Depends(get_config),
    principal: Principal = Depends(require_reviewer_or_admin),
) -> Paginated[PendingCandidateOut]:
    resolved_limit, resolved_offset = resolve_pagination(limit, offset, config.api)
    if principal.role == "admin":
        candidates = repo.list_pending_candidates(limit=resolved_limit, offset=resolved_offset)
        total = repo.count_pending_candidates()
    else:
        if principal.user_id is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Reviewer account has no user_id")
        candidates = repo.list_pending_candidates_for_reviewer(principal.user_id, limit=resolved_limit, offset=resolved_offset)
        total = repo.count_pending_candidates_for_reviewer(principal.user_id)

    items: list[PendingCandidateOut] = []
    for c in candidates:
        student = repo.get_student(c.student_id)
        image = repo.get_pending_candidate_image(c.candidate_id)
        if student is None or image is None:
            continue  # resolved/expired between the list and detail queries -- skip rather than error
        items.append(
            PendingCandidateOut(
                candidate_id=c.candidate_id,
                student_id=c.student_id,
                student_display_name=student.display_name,
                tracker_id=c.tracker_id,
                session_id=c.session_id,
                distance=c.distance,
                captured_at=c.captured_at,
                image_b64=base64.b64encode(image).decode("utf-8"),
            )
        )
    return Paginated(items=items, total=total, limit=resolved_limit, offset=resolved_offset)


@router.post("/{candidate_id}/approve", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
def approve_candidate(
    candidate_id: UUID,
    repo: Repository = Depends(get_repo),
    principal: Principal = Depends(require_reviewer_or_admin),
) -> None:
    """Adds the candidate's embedding to its student's profile
    (source="auto_update") and discards the raw image immediately --
    the photo only ever existed for this review step."""
    resolved = repo.get_pending_candidate_for_approval(candidate_id)
    if resolved is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate not found (already resolved or expired)")

    student_id, vector, quality = resolved
    _assert_reviewer_can_act(repo, principal, student_id)

    embedding = FaceEmbedding(student_id=student_id, vector=vector.tolist(), quality=quality, source="auto_update")
    repo.insert_embedding(embedding)
    repo.delete_pending_candidate(candidate_id)
    repo.flush()

    logger.info(
        "Auto-update candidate approved",
        extra={"context": {"principal": principal.username, "candidate_id": str(candidate_id), "student_id": str(student_id)}},
    )


@router.post("/{candidate_id}/reject", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
def reject_candidate(
    candidate_id: UUID,
    repo: Repository = Depends(get_repo),
    principal: Principal = Depends(require_reviewer_or_admin),
) -> None:
    """Discards the candidate entirely -- image and embedding both. No
    record of a rejected candidate is retained anywhere."""
    resolved = repo.get_pending_candidate_for_approval(candidate_id)
    if resolved is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate not found (already resolved or expired)")

    student_id, _vector, _quality = resolved
    _assert_reviewer_can_act(repo, principal, student_id)

    repo.delete_pending_candidate(candidate_id)
    repo.flush()

    logger.info(
        "Auto-update candidate rejected",
        extra={"context": {"principal": principal.username, "candidate_id": str(candidate_id)}},
    )
