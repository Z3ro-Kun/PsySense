"""
tests/test_api_students.py

Student list/detail/enrollment endpoint tests. Reuses the fake-embedder
fixture setup from test_api_auth.py so enrollment exercises the real
IdentityManager.enroll_student() code path (quality gate, dedup,
persistence) end-to-end through the HTTP layer, without any real ML model.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_api_auth import app_client  # noqa: E402,F401  (fixture)


def _auth_headers(client) -> dict:
    resp = client.post("/api/v1/auth/login", json={"username": "admin", "password": "correct-horse-battery-staple"})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _jpeg_bytes(value: int) -> bytes:
    img = np.full((64, 64, 3), value, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_enroll_student_via_multipart(app_client):
    headers = _auth_headers(app_client)
    resp = app_client.post(
        "/api/v1/students",
        headers=headers,
        data={"display_name": "Alice", "external_ref": "roll-42"},
        files=[("images", ("face1.jpg", io.BytesIO(_jpeg_bytes(200)), "image/jpeg"))],
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["student"]["display_name"] == "Alice"
    assert body["student"]["external_ref"] == "roll-42"
    assert body["accepted_images"] == 1
    assert body["skipped_images"] == []


def test_enroll_requires_at_least_one_image(app_client):
    headers = _auth_headers(app_client)
    resp = app_client.post("/api/v1/students", headers=headers, data={"display_name": "Bob"}, files=[])
    assert resp.status_code in (400, 422)


def test_enroll_requires_auth(app_client):
    resp = app_client.post(
        "/api/v1/students",
        data={"display_name": "Carol"},
        files=[("images", ("f.jpg", io.BytesIO(_jpeg_bytes(100)), "image/jpeg"))],
    )
    assert resp.status_code == 401


def test_list_and_get_student(app_client):
    headers = _auth_headers(app_client)
    enroll = app_client.post(
        "/api/v1/students",
        headers=headers,
        data={"display_name": "Dana"},
        files=[("images", ("f.jpg", io.BytesIO(_jpeg_bytes(50)), "image/jpeg"))],
    )
    student_id = enroll.json()["student"]["student_id"]

    listed = app_client.get("/api/v1/students", headers=headers)
    assert listed.status_code == 200
    body = listed.json()
    assert body["total"] >= 1
    assert any(s["student_id"] == student_id for s in body["items"])

    detail = app_client.get(f"/api/v1/students/{student_id}", headers=headers)
    assert detail.status_code == 200
    assert detail.json()["display_name"] == "Dana"


def test_get_unknown_student_returns_404(app_client):
    headers = _auth_headers(app_client)
    resp = app_client.get("/api/v1/students/00000000-0000-0000-0000-000000000000", headers=headers)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "NOT_FOUND"


def test_list_students_pagination(app_client):
    headers = _auth_headers(app_client)
    for i in range(3):
        app_client.post(
            "/api/v1/students",
            headers=headers,
            data={"display_name": f"Student{i}"},
            files=[("images", (f"f{i}.jpg", io.BytesIO(_jpeg_bytes(30 + i)), "image/jpeg"))],
        )
    resp = app_client.get("/api/v1/students", headers=headers, params={"limit": 2, "offset": 0})
    body = resp.json()
    assert len(body["items"]) == 2
    assert body["limit"] == 2
    assert body["total"] >= 3


def test_delete_student(app_client):
    headers = _auth_headers(app_client)
    enroll = app_client.post(
        "/api/v1/students",
        headers=headers,
        data={"display_name": "Frank"},
        files=[("images", ("f.jpg", io.BytesIO(_jpeg_bytes(77)), "image/jpeg"))],
    )
    student_id = enroll.json()["student"]["student_id"]

    resp = app_client.delete(f"/api/v1/students/{student_id}", headers=headers)
    assert resp.status_code == 204

    # Deleted -- no longer listed or fetchable.
    detail = app_client.get(f"/api/v1/students/{student_id}", headers=headers)
    assert detail.status_code == 404

    listed = app_client.get("/api/v1/students", headers=headers)
    assert not any(s["student_id"] == student_id for s in listed.json()["items"])


def test_delete_unknown_student_returns_404(app_client):
    headers = _auth_headers(app_client)
    resp = app_client.delete("/api/v1/students/00000000-0000-0000-0000-000000000000", headers=headers)
    assert resp.status_code == 404


def test_add_photos_to_existing_student(app_client):
    """The actual fix for the duplicate-profile bug this endpoint was
    built for: a second upload targeting the same student_id must add to
    that profile, not create a second student with the same name."""
    headers = _auth_headers(app_client)
    enroll = app_client.post(
        "/api/v1/students",
        headers=headers,
        data={"display_name": "Mallory", "external_ref": "roll-7"},
        files=[("images", ("f1.jpg", io.BytesIO(_jpeg_bytes(50)), "image/jpeg"))],
    )
    assert enroll.status_code == 201
    student_id = enroll.json()["student"]["student_id"]

    added = app_client.post(
        f"/api/v1/students/{student_id}/photos",
        headers=headers,
        files=[("images", ("f2.jpg", io.BytesIO(_jpeg_bytes(220)), "image/jpeg"))],  # distinct pixel value -- not a duplicate
    )
    assert added.status_code == 200, added.text
    body = added.json()
    assert body["student"]["student_id"] == student_id
    assert body["student"]["display_name"] == "Mallory"  # preserved from original enrollment
    assert body["accepted_images"] == 1

    # Still exactly one student named Mallory -- no duplicate created.
    listed = app_client.get("/api/v1/students", headers=headers, params={"search": "Mallory"})
    assert listed.json()["total"] == 1


def test_add_photos_to_unknown_student_returns_404(app_client):
    headers = _auth_headers(app_client)
    resp = app_client.post(
        "/api/v1/students/00000000-0000-0000-0000-000000000000/photos",
        headers=headers,
        files=[("images", ("f.jpg", io.BytesIO(_jpeg_bytes(90)), "image/jpeg"))],
    )
    assert resp.status_code == 404


def test_add_photos_requires_auth(app_client):
    resp = app_client.post(
        "/api/v1/students/00000000-0000-0000-0000-000000000000/photos",
        files=[("images", ("f.jpg", io.BytesIO(_jpeg_bytes(90)), "image/jpeg"))],
    )
    assert resp.status_code == 401


def test_delete_requires_auth(app_client):
    resp = app_client.delete("/api/v1/students/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 401


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
