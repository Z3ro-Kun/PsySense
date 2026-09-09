"""
tests/test_api_users.py

Sub-admin "reviewer" account tests: creation, login, and -- the actual
point of this feature -- that a reviewer only ever sees/acts on identity
candidates for students they're explicitly assigned to, never the whole
system, and never routes outside the review queue.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models import FaceQualityReport, Student  # noqa: E402
from tests.test_api_auth import app_client  # noqa: E402,F401  (fixture)


def _admin_headers(client) -> dict:
    resp = client.post("/api/v1/auth/login", json={"username": "admin", "password": "correct-horse-battery-staple"})
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _create_reviewer(client, admin_headers, username="teacher1", password="reviewer-pw-123") -> str:
    resp = client.post(
        "/api/v1/users", headers=admin_headers,
        json={"username": username, "password": password, "role": "reviewer"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["user_id"]


def _reviewer_headers(client, username="teacher1", password="reviewer-pw-123") -> dict:
    resp = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _seed_student_with_candidate(client, name: str) -> tuple[str, str]:
    repo = client.app.state.repo
    student = Student(display_name=name)
    repo.upsert_student(student)
    quality = FaceQualityReport(passed=True, blur_score=150.0, brightness=120.0, yaw_deg=1.0, occlusion_score=0.05, face_size_px=100, reasons=[])
    vec = np.random.default_rng(hash(name) % (2**31)).random(8).astype(np.float32)
    candidate_id = repo.insert_pending_candidate(
        student_id=student.student_id, vector=vec, image_jpeg=b"\xff\xd8fake", quality=quality, distance=0.05,
    )
    repo.flush()
    return str(student.student_id), str(candidate_id)


def test_admin_can_create_and_list_reviewer(app_client):
    admin_headers = _admin_headers(app_client)
    _create_reviewer(app_client, admin_headers)

    listed = app_client.get("/api/v1/users", headers=admin_headers)
    assert listed.status_code == 200
    usernames = [u["username"] for u in listed.json()]
    assert "teacher1" in usernames


def test_login_returns_role(app_client):
    admin_headers = _admin_headers(app_client)
    _create_reviewer(app_client, admin_headers)

    reviewer_login = app_client.post("/api/v1/auth/login", json={"username": "teacher1", "password": "reviewer-pw-123"})
    assert reviewer_login.status_code == 200
    assert reviewer_login.json()["role"] == "reviewer"

    admin_login = app_client.post("/api/v1/auth/login", json={"username": "admin", "password": "correct-horse-battery-staple"})
    assert admin_login.json()["role"] == "admin"


def test_reviewer_cannot_access_admin_only_routes(app_client):
    admin_headers = _admin_headers(app_client)
    _create_reviewer(app_client, admin_headers)
    reviewer_headers = _reviewer_headers(app_client)

    assert app_client.get("/api/v1/students", headers=reviewer_headers).status_code == 403
    assert app_client.get("/api/v1/users", headers=reviewer_headers).status_code == 403


def test_deactivated_reviewer_cannot_login(app_client):
    admin_headers = _admin_headers(app_client)
    user_id = _create_reviewer(app_client, admin_headers)

    resp = app_client.post(f"/api/v1/users/{user_id}/deactivate", headers=admin_headers)
    assert resp.status_code == 204

    login = app_client.post("/api/v1/auth/login", json={"username": "teacher1", "password": "reviewer-pw-123"})
    assert login.status_code == 401


def test_duplicate_username_rejected(app_client):
    admin_headers = _admin_headers(app_client)
    _create_reviewer(app_client, admin_headers)
    dup = app_client.post(
        "/api/v1/users", headers=admin_headers,
        json={"username": "teacher1", "password": "another-pw-123", "role": "reviewer"},
    )
    assert dup.status_code == 400


def test_reviewer_sees_only_assigned_candidates(app_client):
    admin_headers = _admin_headers(app_client)
    user_id = _create_reviewer(app_client, admin_headers)
    reviewer_headers = _reviewer_headers(app_client)

    student_a, _cand_a = _seed_student_with_candidate(app_client, "Alice")
    student_b, _cand_b = _seed_student_with_candidate(app_client, "Bob")

    # Assign the reviewer to Alice only.
    assign = app_client.put(
        f"/api/v1/users/{user_id}/assignments", headers=admin_headers, json={"student_ids": [student_a]},
    )
    assert assign.status_code == 200

    as_reviewer = app_client.get("/api/v1/identity-candidates", headers=reviewer_headers)
    assert as_reviewer.status_code == 200
    body = as_reviewer.json()
    assert body["total"] == 1
    assert body["items"][0]["student_id"] == student_a

    as_admin = app_client.get("/api/v1/identity-candidates", headers=admin_headers)
    assert as_admin.json()["total"] == 2


def test_reviewer_cannot_approve_unassigned_candidate(app_client):
    admin_headers = _admin_headers(app_client)
    user_id = _create_reviewer(app_client, admin_headers)
    reviewer_headers = _reviewer_headers(app_client)

    _student_a, cand_a = _seed_student_with_candidate(app_client, "Carol")
    student_b, cand_b = _seed_student_with_candidate(app_client, "Dave")

    app_client.put(f"/api/v1/users/{user_id}/assignments", headers=admin_headers, json={"student_ids": [student_b]})

    # Assigned to Dave -> can approve Dave's candidate.
    approve_ok = app_client.post(f"/api/v1/identity-candidates/{cand_b}/approve", headers=reviewer_headers)
    assert approve_ok.status_code == 204

    # NOT assigned to Carol -> forbidden, even though it's a valid candidate.
    approve_forbidden = app_client.post(f"/api/v1/identity-candidates/{cand_a}/approve", headers=reviewer_headers)
    assert approve_forbidden.status_code == 403


def test_set_assignments_replace_semantics(app_client):
    admin_headers = _admin_headers(app_client)
    user_id = _create_reviewer(app_client, admin_headers)

    student_a, _ = _seed_student_with_candidate(app_client, "Erin")
    student_b, _ = _seed_student_with_candidate(app_client, "Frank")

    app_client.put(f"/api/v1/users/{user_id}/assignments", headers=admin_headers, json={"student_ids": [student_a, student_b]})
    first = app_client.get(f"/api/v1/users/{user_id}/assignments", headers=admin_headers)
    assert sorted(first.json()["student_ids"]) == sorted([student_a, student_b])

    # Replacing with just student_a must drop student_b, not add to it.
    app_client.put(f"/api/v1/users/{user_id}/assignments", headers=admin_headers, json={"student_ids": [student_a]})
    second = app_client.get(f"/api/v1/users/{user_id}/assignments", headers=admin_headers)
    assert second.json()["student_ids"] == [student_a]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
