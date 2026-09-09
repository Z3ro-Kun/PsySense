"""
tests/test_api_auth.py

Auth flow tests against a real (temp-file) SQLite DB but fake embedder --
no real models, no ADMIN_*/JWT_SECRET environment variables required
(AuthSettings is constructed directly and injected via create_app()).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.auth import AuthSettings, hash_password  # noqa: E402
from api.main import create_app  # noqa: E402
from core.config import AppConfig  # noqa: E402
from db.repository.repository import Repository  # noqa: E402
from db.writer import AsyncDBWriter  # noqa: E402
from services.identity.cache import ActiveIdentityCache  # noqa: E402
from services.identity.manager import IdentityManager  # noqa: E402
from services.identity.vector_index import LinearVectorIndex  # noqa: E402
from tests.test_identity_manager import FakeEmbedder, FakeQualityAssessor  # noqa: E402

SCHEMA_PATH = str(Path(__file__).resolve().parents[1] / "db" / "schema.sql")


def _fake_auth_settings() -> AuthSettings:
    settings = AuthSettings.__new__(AuthSettings)  # bypass __init__'s env-var requirement
    settings.admin_username = "admin"
    settings.admin_password_hash = hash_password("correct-horse-battery-staple")
    settings.jwt_secret = "test-secret-not-for-production"
    return settings


@pytest.fixture()
def app_client(tmp_path):
    db_path = str(tmp_path / "api_test.db")
    writer = AsyncDBWriter(db_path=db_path, schema_path=SCHEMA_PATH)
    writer.start()
    time.sleep(0.1)
    repo = Repository(writer=writer, db_path=db_path)

    embedder = FakeEmbedder()
    identity_manager = IdentityManager(
        embedder=embedder,
        quality_assessor=FakeQualityAssessor(always_pass=True),
        vector_index=LinearVectorIndex(dim=embedder.dim),
        cache=ActiveIdentityCache(),
        repository=repo,
    )

    app = create_app(
        config=AppConfig(),
        repository=repo,
        identity_manager=identity_manager,
        auth_settings=_fake_auth_settings(),
    )
    with TestClient(app) as client:
        yield client
    repo.close()
    writer.stop()


def test_login_succeeds_with_correct_credentials(app_client):
    resp = app_client.post("/api/v1/auth/login", json={"username": "admin", "password": "correct-horse-battery-staple"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]


def test_login_fails_with_wrong_password(app_client):
    resp = app_client.post("/api/v1/auth/login", json={"username": "admin", "password": "wrong"})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "UNAUTHORIZED"


def test_protected_route_rejects_missing_token(app_client):
    resp = app_client.get("/api/v1/students")
    assert resp.status_code == 401


def test_protected_route_rejects_garbage_token(app_client):
    resp = app_client.get("/api/v1/students", headers={"Authorization": "Bearer not-a-real-token"})
    assert resp.status_code == 401


def test_protected_route_accepts_valid_token(app_client):
    login = app_client.post("/api/v1/auth/login", json={"username": "admin", "password": "correct-horse-battery-staple"})
    token = login.json()["access_token"]
    resp = app_client.get("/api/v1/students", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200


def test_login_lockout_after_repeated_failures(app_client):
    for _ in range(5):
        app_client.post("/api/v1/auth/login", json={"username": "admin", "password": "wrong"})
    resp = app_client.post("/api/v1/auth/login", json={"username": "admin", "password": "correct-horse-battery-staple"})
    assert resp.status_code == 429


def test_health_and_ready_are_public(app_client):
    assert app_client.get("/health").status_code == 200
    assert app_client.get("/ready").status_code == 200


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
