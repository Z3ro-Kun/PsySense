"""
api/routers/users.py

Admin-only account management for sub-admin "reviewer" accounts (see
db/schema.sql's users/reviewer_assignments tables and
core/models.py's User/UserRole). No self-signup: only an existing admin
can create an account here, and only for someone they set an initial
password for directly -- consistent with how the root admin's own
credentials are generated (scripts/create_admin.py).
"""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from api.auth import hash_password
from api.deps import get_repo, require_admin
from api.schemas import AssignmentsOut, AssignmentsRequest, CreateUserRequest, UserOut
from db.repository.repository import Repository
from logging_.logger import get_logger

logger = get_logger("api.users")

router = APIRouter(prefix="/api/v1/users", tags=["users"], dependencies=[Depends(require_admin)])


@router.get("", response_model=list[UserOut])
def list_users(repo: Repository = Depends(get_repo)) -> list[UserOut]:
    users = repo.list_users()
    return [
        UserOut(**u.model_dump(), assigned_student_count=repo.count_reviewer_assignments(u.user_id))
        for u in users
    ]


@router.post("", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user(
    body: CreateUserRequest,
    repo: Repository = Depends(get_repo),
    admin: str = Depends(require_admin),
) -> UserOut:
    if repo.get_user_by_username(body.username) is not None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already taken")
    if len(body.password) < 8:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password must be at least 8 characters")

    user_id = repo.create_user(body.username, hash_password(body.password), role=body.role)
    repo.flush()
    user = repo.get_user(user_id)
    logger.info(
        "User account created",
        extra={"context": {"admin": admin, "new_user": body.username, "role": body.role.value}},
    )
    return UserOut(**user.model_dump(), assigned_student_count=0)


@router.post("/{user_id}/deactivate", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
def deactivate_user(
    user_id: UUID,
    repo: Repository = Depends(get_repo),
    admin: str = Depends(require_admin),
) -> None:
    if repo.get_user(user_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    repo.set_user_active(user_id, False)
    repo.flush()
    logger.info("User account deactivated", extra={"context": {"admin": admin, "user_id": str(user_id)}})


@router.post("/{user_id}/reactivate", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
def reactivate_user(
    user_id: UUID,
    repo: Repository = Depends(get_repo),
    admin: str = Depends(require_admin),
) -> None:
    if repo.get_user(user_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    repo.set_user_active(user_id, True)
    repo.flush()
    logger.info("User account reactivated", extra={"context": {"admin": admin, "user_id": str(user_id)}})


@router.get("/{user_id}/assignments", response_model=AssignmentsOut)
def get_assignments(user_id: UUID, repo: Repository = Depends(get_repo)) -> AssignmentsOut:
    if repo.get_user(user_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return AssignmentsOut(student_ids=repo.get_reviewer_assignments(user_id))


@router.put("/{user_id}/assignments", response_model=AssignmentsOut)
def set_assignments(
    user_id: UUID,
    body: AssignmentsRequest,
    repo: Repository = Depends(get_repo),
    admin: str = Depends(require_admin),
) -> AssignmentsOut:
    """Replace semantics -- the reviewer's assigned-students set becomes
    exactly body.student_ids."""
    if repo.get_user(user_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    for student_id in body.student_ids:
        if repo.get_student(student_id) is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown student_id: {student_id}")

    repo.set_reviewer_assignments(user_id, body.student_ids)
    repo.flush()
    logger.info(
        "Reviewer assignments updated",
        extra={"context": {"admin": admin, "user_id": str(user_id), "count": len(body.student_ids)}},
    )
    return AssignmentsOut(student_ids=body.student_ids)
