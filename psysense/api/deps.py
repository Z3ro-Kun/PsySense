"""
api/deps.py

FastAPI dependencies. All state (Repository, AuthSettings, login tracker,
config) lives on app.state, set once by api/main.py's create_app() factory
-- no module-level globals, so tests can build multiple independent app
instances (e.g. pointed at different temp databases) in the same process.
"""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.auth import AuthSettings, FailedLoginTracker, Principal, decode_access_token
from core.config import AppConfig
from db.repository.repository import Repository

_bearer_scheme = HTTPBearer(auto_error=False)


def get_config(request: Request) -> AppConfig:
    return request.app.state.config


def get_repo(request: Request) -> Repository:
    return request.app.state.repo


def get_auth_settings(request: Request) -> AuthSettings:
    return request.app.state.auth_settings


def get_login_tracker(request: Request) -> FailedLoginTracker:
    return request.app.state.login_tracker


def get_identity_manager(request: Request):
    """Returns the API process's own IdentityManager instance, used only
    for enroll_student() (quality gating + dedup + persistence). It does
    NOT do face resolution/search -- that stays the responsibility of the
    running pipeline process, which picks up new enrollments via
    IdentityManager.refresh_from_repository() (see main.py)."""
    return request.app.state.identity_manager


def _authenticate(request: Request, credentials: Optional[HTTPAuthorizationCredentials]) -> Principal:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    auth_settings: AuthSettings = request.app.state.auth_settings
    principal = decode_access_token(credentials.credentials, auth_settings.jwt_secret)
    if principal is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    return principal


def require_admin(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> str:
    """Validates the Bearer JWT, requires role="admin", and returns the
    admin's username, else 401/403. Every admin-only route depends on
    this (students, delete, health/ready/login are the exceptions --
    see api/main.py's router wiring). Reviewer-role tokens are rejected
    here -- use require_reviewer_or_admin for routes reviewers may access."""
    principal = _authenticate(request, credentials)
    if principal.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return principal.username


def require_reviewer_or_admin(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Principal:
    """For routes both roles may access (currently just the identity
    candidates review queue) -- returns the full Principal since the
    route itself must further scope by principal.role/user_id (a reviewer
    only sees/acts on their assigned students, an admin sees everyone)."""
    return _authenticate(request, credentials)
