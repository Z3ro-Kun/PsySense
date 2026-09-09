"""
api/routers/students.py

List/detail/enroll endpoints. Enrollment reuses
services.identity.manager.IdentityManager.enroll_student() as-is (quality
gating, near-duplicate rejection, DB persistence are all already
implemented there) rather than re-implementing any of that here -- this
router's only job is turning multipart form data into the arguments that
method expects, and validating upload limits before decoding anything.
"""
from __future__ import annotations

from typing import Optional
from uuid import UUID

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from api.deps import get_config, get_identity_manager, get_repo, require_admin
from api.pagination import resolve_pagination
from api.schemas import EnrollmentResult, Paginated, StudentOut
from core.config import AppConfig
from db.repository.repository import Repository
from logging_.logger import get_logger
from services.identity.manager import IdentityManager

logger = get_logger("api.students")

router = APIRouter(prefix="/api/v1/students", tags=["students"])

MAX_IMAGES_PER_ENROLLMENT = 10
MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8MB/image -- enrollment photos, not video


@router.get("", response_model=Paginated[StudentOut])
def list_students(
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    search: Optional[str] = None,
    repo: Repository = Depends(get_repo),
    config: AppConfig = Depends(get_config),
    _admin: str = Depends(require_admin),
) -> Paginated[StudentOut]:
    resolved_limit, resolved_offset = resolve_pagination(limit, offset, config.api)
    students = repo.list_students(limit=resolved_limit, offset=resolved_offset, search=search)
    total = repo.count_students(search=search)
    return Paginated(
        items=[StudentOut(**s.model_dump()) for s in students],
        total=total, limit=resolved_limit, offset=resolved_offset,
    )


@router.get("/{student_id}", response_model=StudentOut)
def get_student(
    student_id: UUID,
    repo: Repository = Depends(get_repo),
    _admin: str = Depends(require_admin),
) -> StudentOut:
    student = repo.get_student(student_id)
    if student is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
    return StudentOut(**student.model_dump())


async def _decode_upload_images(images: list[UploadFile]) -> list[np.ndarray]:
    if not images:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one image is required")
    if len(images) > MAX_IMAGES_PER_ENROLLMENT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many images (max {MAX_IMAGES_PER_ENROLLMENT} per request)",
        )

    decoded: list[np.ndarray] = []
    for upload in images:
        raw = await upload.read()
        if len(raw) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image '{upload.filename}' exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)}MB limit",
            )
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not decode image '{upload.filename}'",
            )
        decoded.append(frame)
    return decoded


@router.post("", response_model=EnrollmentResult, status_code=status.HTTP_201_CREATED)
async def enroll_student(
    display_name: str = Form(...),
    external_ref: Optional[str] = Form(None),
    images: list[UploadFile] = File(...),
    identity_manager: IdentityManager = Depends(get_identity_manager),
    repo: Repository = Depends(get_repo),
    admin: str = Depends(require_admin),
) -> EnrollmentResult:
    decoded = await _decode_upload_images(images)

    student, accepted, skipped = identity_manager.enroll_student(
        display_name=display_name, images_bgr=decoded, external_ref=external_ref,
    )
    # enroll_student()'s DB writes are fire-and-forget (AsyncDBWriter);
    # block here until they've landed so a client's immediate follow-up
    # GET /students reliably sees the new record (see db/writer.py's
    # barrier() docstring).
    repo.flush()
    logger.info(
        "Enrollment via API",
        extra={"context": {"admin": admin, "student_id": str(student.student_id),
                            "accepted": len(accepted), "skipped": len(skipped)}},
    )
    return EnrollmentResult(
        student=StudentOut(**student.model_dump()),
        accepted_images=len(accepted),
        skipped_images=skipped,
    )


@router.post("/{student_id}/photos", response_model=EnrollmentResult)
async def add_photos(
    student_id: UUID,
    images: list[UploadFile] = File(...),
    identity_manager: IdentityManager = Depends(get_identity_manager),
    repo: Repository = Depends(get_repo),
    admin: str = Depends(require_admin),
) -> EnrollmentResult:
    """Adds more embeddings to an EXISTING student -- distinct from
    POST /students, which always creates a new one. This is the actual
    fix for weak single-photo profiles: enroll_student(student_id=...)
    adds to the same profile's embedding history instead of creating a
    duplicate student, which is what silently happened before this
    endpoint existed (see the "Vinay"/"Vinay Tiwari" duplicate-profile
    incident this was built to fix)."""
    existing = repo.get_student(student_id)
    if existing is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")

    decoded = await _decode_upload_images(images)

    student, accepted, skipped = identity_manager.enroll_student(
        display_name=existing.display_name, images_bgr=decoded,
        external_ref=existing.external_ref, student_id=student_id,
    )
    repo.flush()
    logger.info(
        "Photos added via API",
        extra={"context": {"admin": admin, "student_id": str(student_id),
                            "accepted": len(accepted), "skipped": len(skipped)}},
    )
    return EnrollmentResult(
        student=StudentOut(**student.model_dump()),
        accepted_images=len(accepted),
        skipped_images=skipped,
    )


@router.delete("/{student_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
def delete_student(
    student_id: UUID,
    identity_manager: IdentityManager = Depends(get_identity_manager),
    repo: Repository = Depends(get_repo),
    admin: str = Depends(require_admin),
) -> None:
    """Hard delete (see Repository.delete_student()'s docstring for why).
    Also removes the student from this process's own IdentityManager so a
    subsequent enroll_student(student_id=...) in the same process doesn't
    treat them as still existing; the separate, authoritative recognition
    instance running in main.py picks up the deletion on its next
    refresh_from_repository() cycle."""
    student = repo.get_student(student_id)
    if student is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")

    repo.delete_student(student_id)
    repo.flush()
    identity_manager.remove_student(student_id)

    logger.info("Student deleted via API", extra={"context": {"admin": admin, "student_id": str(student_id)}})
