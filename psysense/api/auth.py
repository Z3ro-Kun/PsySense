"""
api/auth.py

Authentication for bridge_api: a single root admin account from
ADMIN_USERNAME/ADMIN_PASSWORD_HASH/JWT_SECRET (env vars only, never
committed -- use scripts/create_admin.py to generate the hash), plus an
optional layer of sub-admin "reviewer" accounts stored in the users table
(db/schema.sql) for reviewing identity-update candidates scoped to
specific students (see api/routers/users.py and
reviewer_assignments). This is still deliberately not a full multi-tenant
system (no organizations, no per-org isolation) -- see CLAUDE.md Sec 37
and the psysense-multitenant-direction memory for that larger, explicitly
deferred decision.

AuthSettings fails loudly if the root admin env vars are missing -- the
same "fail at startup, not with a confusing 500 later" rule core/config.py
already applies to config.yaml.

Uses the `bcrypt` package directly rather than passlib: passlib 1.7.4 is
unmaintained upstream (last release 2020) and its bcrypt backend probes
`bcrypt.__about__.__version__`, an attribute modern bcrypt (>=4.1) no
longer exposes -- it still works (passlib traps the error) but logs a
spurious warning on every process start. Talking to bcrypt directly avoids
depending on an abandoned compatibility shim for something this small.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

import bcrypt
import jwt


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. bridge_api requires ADMIN_USERNAME, ADMIN_PASSWORD_HASH, "
            f"and JWT_SECRET as environment variables -- run "
            f"`python -m scripts.create_admin` to generate a password hash. "
            f"Refusing to start without explicit credentials."
        )
    return value


class AuthSettings:
    """Reads and validates required env vars once, at app-startup time (see
    api/main.py's create_app), not per-request."""

    def __init__(self) -> None:
        self.admin_username: str = _require_env("ADMIN_USERNAME")
        self.admin_password_hash: str = _require_env("ADMIN_PASSWORD_HASH")
        self.jwt_secret: str = _require_env("JWT_SECRET")


class FailedLoginTracker:
    """In-memory per-username lockout after N consecutive failures within a
    window. Not persisted (resets on API restart) and not distributed --
    a lightweight guard against brute-forcing the one admin password, not
    a full rate limiter. Proportionate to a single-admin deployment;
    revisit if this ever becomes multi-user."""

    def __init__(self, max_failures: int, lockout_seconds: float):
        self._max_failures = max_failures
        self._lockout_seconds = lockout_seconds
        self._failures: dict[str, list[float]] = {}

    def is_locked_out(self, username: str) -> bool:
        now = time.time()
        recent = [t for t in self._failures.get(username, []) if now - t < self._lockout_seconds]
        self._failures[username] = recent
        return len(recent) >= self._max_failures

    def record_failure(self, username: str) -> None:
        self._failures.setdefault(username, []).append(time.time())

    def record_success(self, username: str) -> None:
        self._failures.pop(username, None)


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        # Malformed hash in ADMIN_PASSWORD_HASH -- treat as a failed
        # verification, not a 500, but this should be caught at startup
        # by a smoke test/health check, not discovered on first login.
        return False


@dataclass(frozen=True)
class Principal:
    """The authenticated caller: either the root admin (role="admin",
    user_id=None) or a users-table account (role="admin"|"reviewer",
    user_id set). Routes that need to scope data to a specific reviewer
    (api/routers/identity_candidates.py) use user_id; require_admin
    (api/deps.py) only ever checks role."""

    username: str
    role: str
    user_id: Optional[UUID] = None


def create_access_token(
    subject: str, secret: str, expire_minutes: int, role: str = "admin", user_id: Optional[UUID] = None,
) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject, "role": role, "user_id": str(user_id) if user_id else None,
        "iat": now, "exp": now + timedelta(minutes=expire_minutes),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_access_token(token: str, secret: str) -> Optional[Principal]:
    """Returns the token's Principal if valid, else None -- never raises,
    so callers can turn any failure into a uniform 401. Tokens issued
    before the role/user_id claims existed have no "role" key -- treated
    as role="admin" for backward compatibility with already-issued root
    admin tokens (the only kind that existed before this)."""
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
    sub = payload.get("sub")
    if not sub:
        return None
    user_id_raw = payload.get("user_id")
    return Principal(
        username=sub,
        role=payload.get("role", "admin"),
        user_id=UUID(user_id_raw) if user_id_raw else None,
    )
