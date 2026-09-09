"""
tests/test_api_telemetry.py

Session/emotion/pose/behavior-event read endpoints: pagination, time-range
filtering, and student scoping. Writes go straight through the Repository
(bypassing HTTP) to set up fixtures, then reads go through the API.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models import Student  # noqa: E402
from tests.test_api_auth import app_client  # noqa: E402,F401  (fixture)


def _auth_headers(client) -> dict:
    resp = client.post("/api/v1/auth/login", json={"username": "admin", "password": "correct-horse-battery-staple"})
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _seed(client) -> str:
    """Writes directly through the Repository the app was built with
    (accessible via the test app's state) to avoid depending on the
    enrollment HTTP path for fixtures unrelated to enrollment itself."""
    repo = client.app.state.repo
    student = Student(display_name="Erin")
    repo.upsert_student(student)
    sid = str(student.student_id)
    repo.open_session("sess-1", tracker_id=1, student_id=student.student_id)
    repo.insert_emotion_event("sess-1", student.student_id, {"happy": 0.9, "sad": 0.1})
    repo.insert_pose_event("sess-1", student.student_id, 0.8, 0.2, 0.1)
    repo.insert_behavior_event("sess-1", student.student_id, "prolonged_slump", {"detail": "x"}, severity="warning")
    repo.flush()
    return sid


def test_list_sessions(app_client):
    headers = _auth_headers(app_client)
    _seed(app_client)
    resp = app_client.get("/api/v1/sessions", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["items"][0]["tracker_id"] == 1
    assert body["items"][0]["status"] == "active"


def test_student_emotions_scoped_correctly(app_client):
    headers = _auth_headers(app_client)
    sid = _seed(app_client)
    other_id = "00000000-0000-0000-0000-000000000000"

    mine = app_client.get(f"/api/v1/students/{sid}/emotions", headers=headers)
    assert mine.status_code == 200
    assert mine.json()["total"] == 1
    assert mine.json()["items"][0]["dominant_emotion"] == "happy"

    other = app_client.get(f"/api/v1/students/{other_id}/emotions", headers=headers)
    assert other.status_code == 200
    assert other.json()["total"] == 0  # must never leak another student's events


def test_student_pose_events(app_client):
    headers = _auth_headers(app_client)
    sid = _seed(app_client)
    resp = app_client.get(f"/api/v1/students/{sid}/pose", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["items"][0]["slumped_score"] == 0.8


def test_behavior_events_global_and_scoped(app_client):
    headers = _auth_headers(app_client)
    sid = _seed(app_client)

    global_events = app_client.get("/api/v1/behavior-events", headers=headers)
    assert global_events.status_code == 200
    assert global_events.json()["total"] == 1

    scoped = app_client.get(f"/api/v1/students/{sid}/behavior-events", headers=headers)
    assert scoped.status_code == 200
    assert scoped.json()["items"][0]["event_type"] == "prolonged_slump"
    assert scoped.json()["items"][0]["severity"] == "warning"

    filtered = app_client.get("/api/v1/behavior-events", headers=headers, params={"severity": "critical"})
    assert filtered.json()["total"] == 0


def test_pagination_limit_is_clamped_to_max_page_size(app_client):
    headers = _auth_headers(app_client)
    _seed(app_client)
    resp = app_client.get("/api/v1/sessions", headers=headers, params={"limit": 999999})
    assert resp.status_code == 200
    assert resp.json()["limit"] <= 200  # ApiConfig.max_page_size default


def test_telemetry_requires_auth(app_client):
    resp = app_client.get("/api/v1/sessions")
    assert resp.status_code == 401


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
