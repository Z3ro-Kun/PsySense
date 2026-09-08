"""
tests/test_identity_manager.py

Unit tests for IdentityManager using fakes for the embedder, quality
assessor, and vector index — these tests validate orchestration logic
(caching, matching, duplicate rejection, tracker-ID decoupling) without
needing InsightFace models, FAISS, or a camera.
"""
from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models import FaceQualityReport, MatchStatus  # noqa: E402
from services.identity.cache import ActiveIdentityCache  # noqa: E402
from services.identity.manager import IdentityManager  # noqa: E402
from services.identity.vector_index import LinearVectorIndex  # noqa: E402


class FakeQualityAssessor:
    def __init__(self, always_pass: bool = True):
        self.always_pass = always_pass

    def assess(self, face_bgr, face_box, landmarks_5pt=None) -> FaceQualityReport:
        return FaceQualityReport(
            passed=self.always_pass,
            blur_score=200.0,
            brightness=120.0,
            yaw_deg=0.0,
            occlusion_score=0.0,
            face_size_px=100,
            reasons=[] if self.always_pass else ["blur"],
        )


class FakeEmbedder:
    """Deterministic fake: encodes a scalar tag baked into the image's mean
    pixel value into a fixed direction so we can control similarity in
    tests without a real model."""

    dim = 8

    def analyze(self, face_bgr):
        return None  # no landmarks needed for these tests

    def embed(self, face_bgr: np.ndarray):
        if face_bgr is None or face_bgr.size == 0:
            return None
        tag = float(np.mean(face_bgr)) / 255.0
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[0] = tag
        vec[1] = 1.0 - tag
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


def make_manager(match_threshold=0.42, dup_threshold=0.08):
    embedder = FakeEmbedder()
    return IdentityManager(
        embedder=embedder,
        quality_assessor=FakeQualityAssessor(always_pass=True),
        vector_index=LinearVectorIndex(dim=embedder.dim),
        cache=ActiveIdentityCache(ttl_seconds=5.0, reverify_every_n_frames=1000),
        match_distance_threshold=match_threshold,
        duplicate_distance_threshold=dup_threshold,
    )


def _solid_image(value: int, size=(64, 64, 3)) -> np.ndarray:
    return np.full(size, value, dtype=np.uint8)


def test_enroll_and_match_same_person():
    mgr = make_manager()
    student, accepted, skipped = mgr.enroll_student("Alice", [_solid_image(200)])
    assert len(accepted) == 1
    assert skipped == []

    result = mgr.resolve(tracker_id=1, face_bgr=_solid_image(200), face_box=(0, 0, 64, 64))
    assert result.status == MatchStatus.MATCHED
    assert result.student_id == student.student_id


def test_unmatched_face_returns_unknown():
    mgr = make_manager()
    mgr.enroll_student("Alice", [_solid_image(200)])

    # Very different pixel value -> very different fake embedding direction
    result = mgr.resolve(tracker_id=2, face_bgr=_solid_image(10), face_box=(0, 0, 64, 64))
    assert result.status == MatchStatus.UNKNOWN


def test_identity_decoupled_from_tracker_id():
    """Same person, two different tracker IDs (e.g. re-entered frame after
    occlusion) must resolve to the SAME student_id — this is the core
    architectural requirement being validated."""
    mgr = make_manager()
    student, _, _ = mgr.enroll_student("Alice", [_solid_image(200)])

    r1 = mgr.resolve(tracker_id=10, face_bgr=_solid_image(200), face_box=(0, 0, 64, 64))
    mgr.on_tracker_dropped(10)  # simulate occlusion / person leaving frame
    r2 = mgr.resolve(tracker_id=57, face_bgr=_solid_image(200), face_box=(0, 0, 64, 64))  # new tracker id

    assert r1.status == MatchStatus.MATCHED
    assert r2.status == MatchStatus.MATCHED
    assert r1.student_id == r2.student_id == student.student_id


def test_cache_hit_on_second_lookup_same_tracker():
    mgr = make_manager()
    mgr.enroll_student("Alice", [_solid_image(200)])

    first = mgr.resolve(tracker_id=3, face_bgr=_solid_image(200), face_box=(0, 0, 64, 64))
    second = mgr.resolve(tracker_id=3, face_bgr=_solid_image(200), face_box=(0, 0, 64, 64))

    assert first.status == MatchStatus.MATCHED
    assert second.status == MatchStatus.CACHE_HIT
    assert second.student_id == first.student_id


def test_duplicate_enrollment_image_is_skipped():
    mgr = make_manager()
    _student, accepted1, _ = mgr.enroll_student("Alice", [_solid_image(200)])
    student, accepted2, skipped2 = mgr.enroll_student(
        "Alice", [_solid_image(200)], student_id=_student.student_id
    )
    assert len(accepted1) == 1
    assert len(accepted2) == 0
    assert any("duplicate_of" in s for s in skipped2)


def test_quality_rejection_prevents_matching_and_enrollment():
    embedder = FakeEmbedder()
    mgr = IdentityManager(
        embedder=embedder,
        quality_assessor=FakeQualityAssessor(always_pass=False),
        vector_index=LinearVectorIndex(dim=embedder.dim),
    )
    student, accepted, skipped = mgr.enroll_student("Bob", [_solid_image(150)])
    assert accepted == []
    assert len(skipped) == 1

    result = mgr.resolve(tracker_id=1, face_bgr=_solid_image(150), face_box=(0, 0, 64, 64))
    assert result.status == MatchStatus.REJECTED_QUALITY


def test_multiple_students_disambiguated_correctly():
    mgr = make_manager()
    alice, _, _ = mgr.enroll_student("Alice", [_solid_image(220)])
    bob, _, _ = mgr.enroll_student("Bob", [_solid_image(30)])

    r_alice = mgr.resolve(tracker_id=1, face_bgr=_solid_image(220), face_box=(0, 0, 64, 64))
    r_bob = mgr.resolve(tracker_id=2, face_bgr=_solid_image(30), face_box=(0, 0, 64, 64))

    assert r_alice.student_id == alice.student_id
    assert r_bob.student_id == bob.student_id
    assert r_alice.student_id != r_bob.student_id


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))