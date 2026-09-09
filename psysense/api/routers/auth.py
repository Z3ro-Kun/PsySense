"""api/routers/auth.py -- login endpoint. See api/auth.py for the
credential/token/lockout logic this wraps."""
from __future__ import annotations

from typing import NoReturn

from fastapi import APIRouter, Depends, HTTPException, status

from api.auth import AuthSettings, FailedLoginTracker, create_access_token, verify_password
from api.deps import get_auth_settings, get_config, get_login_tracker, get_repo
from api.schemas import LoginRequest, TokenResponse
from core.config import AppConfig
from db.repository.repository import Repository
from logging_.logger import get_logger

logger = get_logger("api.auth_router")

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(
    body: LoginRequest,
    auth_settings: AuthSettings = Depends(get_auth_settings),
    tracker: FailedLoginTracker = Depends(get_login_tracker),
    config: AppConfig = Depends(get_config),
    repo: Repository = Depends(get_repo),
) -> TokenResponse:
    if tracker.is_locked_out(body.username):
        logger.warning("Login blocked by lockout", extra={"context": {"username": body.username}})
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed login attempts. Try again shortly.",
        )

    # Root admin (env vars) is checked first and always wins on a
    # username match -- it's the break-glass account and must keep
    # working even if the users table is empty or corrupted.
    if body.username == auth_settings.admin_username:
        if not verify_password(body.password, auth_settings.admin_password_hash):
            _reject(tracker, body.username)
        tracker.record_success(body.username)
        token = create_access_token(
            body.username, auth_settings.jwt_secret, config.api.access_token_expire_minutes, role="admin",
        )
        logger.info("Login succeeded (root admin)", extra={"context": {"username": body.username}})
        return TokenResponse(access_token=token, expires_in_minutes=config.api.access_token_expire_minutes, role="admin")

    # Otherwise check the users table (sub-admin / reviewer accounts).
    found = repo.get_user_by_username(body.username)
    if found is None:
        _reject(tracker, body.username)
    user, password_hash = found
    if not user.active or not verify_password(body.password, password_hash):
        _reject(tracker, body.username)

    tracker.record_success(body.username)
    token = create_access_token(
        body.username, auth_settings.jwt_secret, config.api.access_token_expire_minutes,
        role=user.role.value, user_id=user.user_id,
    )
    logger.info("Login succeeded", extra={"context": {"username": body.username, "role": user.role.value}})
    return TokenResponse(access_token=token, expires_in_minutes=config.api.access_token_expire_minutes, role=user.role.value)


def _reject(tracker: FailedLoginTracker, username: str) -> NoReturn:
    tracker.record_failure(username)
    logger.warning("Login failed", extra={"context": {"username": username}})
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")
