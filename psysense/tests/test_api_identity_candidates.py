"""
tests/test_api_identity_candidates.py

API-level tests for the review-queue endpoints. Candidates are seeded
directly through the Repository (bypassing IdentityManager.resolve(),
which isn't exercised over HTTP) to isolate what this router itself is
responsible for: listing with the image inlined, and approve/reject
correctly mutating face_embeddings/pending_identity_candidates.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models import FaceQualityReport, Student  # noqa: E402
from tests.test_api_auth import app_client  # noqa: E402,F401  (fixture)


def _auth_headers(client) -> dict:
    resp = client.post("/api/v1/auth/login", json={"username": "admin", "password": "correct-horse-battery-staple"})
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _seed_candidate(client) -> tuple[str, str]:
    repo = client.app.state.repo
    student = Student(display_name="Nadia")
    repo.upsert_student(student)
    quality = FaceQualityReport(
        passed=True, blur_score=150.0, brightness=120.0, yaw_deg=1.0,
        occlusion_score=0.05, face_size_px=100, reasons=[],
    )
    vec = np.random.default_rng(1).random(8).astype(np.float32)
    candidate_id = repo.insert_pending_candidate(
        student_id=student.student_id, vector=vec, image_jpeg=b"\xff\xd8fake-jpeg-bytes",
        quality=quality, distance=0.05, tracker_id=1, session_id=None,
    )
    repo.flush()
    return str(student.student_id), str(candidate_id)


def test_list_candidates_includes_inlined_image(app_client):
    headers = _auth_headers(app_client)
    student_id, candidate_id = _seed_candidate(app_client)

    resp = app_client.get("/api/v1/identity-candidates", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    item = body["items"][0]
    assert item["candidate_id"] == candidate_id
    assert item["student_id"] == student_id
    assert item["student_display_name"] == "Nadia"
    assert len(item["image_b64"]) > 0


def test_approve_candidate_adds_embedding_and_removes_pending(app_client):
    headers = _auth_headers(app_client)
    student_id, candidate_id = _seed_candidate(app_client)

    resp = app_client.post(f"/api/v1/identity-candidates/{candidate_id}/approve", headers=headers)
    assert resp.status_code == 204

    listed = app_client.get("/api/v1/identity-candidates", headers=headers)
    assert listed.json()["total"] == 0

    repo = app_client.app.state.repo
    embeddings = repo.get_all_embeddings()
    assert len(embeddings) == 1
    assert embeddings[0].source == "auto_update"
    assert str(embeddings[0].student_id) == student_id


def test_reject_candidate_adds_no_embedding(app_client):
    headers = _auth_headers(app_client)
    _student_id, candidate_id = _seed_candidate(app_client)

    resp = app_client.post(f"/api/v1/identity-candidates/{candidate_id}/reject", headers=headers)
    assert resp.status_code == 204

    listed = app_client.get("/api/v1/identity-candidates", headers=headers)
    assert listed.json()["total"] == 0

    repo = app_client.app.state.repo
    assert len(repo.get_all_embeddings()) == 0


def test_approve_unknown_candidate_returns_404(app_client):
    headers = _auth_headers(app_client)
    resp = app_client.post("/api/v1/identity-candidates/00000000-0000-0000-0000-000000000000/approve", headers=headers)
    assert resp.status_code == 404


def test_candidates_require_auth(app_client):
    resp = app_client.get("/api/v1/identity-candidates")
    assert resp.status_code == 401


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
