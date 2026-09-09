"""
tests/test_auto_update_candidates.py

Human-in-the-loop identity auto-update: a high-confidence resolve() match
queues a candidate (raw image + embedding) for admin review instead of
ever touching a profile automatically. Covers the pipeline-side queuing
(rate limiting included) and the repository-side approve/reject/cleanup
operations the API layer builds on.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models import MatchStatus  # noqa: E402
from db.repository.repository import Repository  # noqa: E402
from db.writer import AsyncDBWriter  # noqa: E402
from services.identity.cache import ActiveIdentityCache  # noqa: E402
from services.identity.manager import IdentityManager  # noqa: E402
from services.identity.vector_index import LinearVectorIndex  # noqa: E402
from tests.test_identity_manager import FakeEmbedder, FakeQualityAssessor, _solid_image  # noqa: E402

SCHEMA_PATH = str(Path(__file__).resolve().parents[1] / "db" / "schema.sql")


@pytest.fixture()
def db_path(tmp_path):
    return str(tmp_path / "auto_update_test.db")


@pytest.fixture()
def writer(db_path):
    w = AsyncDBWriter(db_path=db_path, schema_path=SCHEMA_PATH)
    w.start()
    yield w
    w.stop()


@pytest.fixture()
def repo(writer, db_path):
    time.sleep(0.1)
    r = Repository(writer=writer, db_path=db_path)
    yield r
    r.close()


def _manager_with_auto_update(repo, **overrides):
    embedder = FakeEmbedder()
    kwargs = dict(
        embedder=embedder,
        quality_assessor=FakeQualityAssessor(always_pass=True),
        vector_index=LinearVectorIndex(dim=embedder.dim),
        cache=ActiveIdentityCache(),
        repository=repo,
        auto_update_enabled=True,
        auto_update_distance_threshold=0.05,  # FakeEmbedder gives ~0 distance on an exact repeat image
        auto_update_min_seconds_between_candidates=3600.0,
    )
    kwargs.update(overrides)
    return IdentityManager(**kwargs)


def test_high_confidence_match_queues_candidate_not_embedding(repo):
    mgr = _manager_with_auto_update(repo)
    student, _, _ = mgr.enroll_student("Grace", [_solid_image(160)])
    time.sleep(0.2)

    result = mgr.resolve(tracker_id=1, face_bgr=_solid_image(160), face_box=(0, 0, 64, 64))
    assert result.status == MatchStatus.MATCHED
    time.sleep(0.2)

    candidates = repo.list_pending_candidates()
    assert len(candidates) == 1
    assert candidates[0].student_id == student.student_id

    # Must NOT have been added directly to the profile -- still just the
    # one original enrollment embedding until a human approves.
    assert len(repo.get_all_embeddings()) == 1


def test_auto_update_rate_limited_per_student(repo):
    mgr = _manager_with_auto_update(repo, auto_update_min_seconds_between_candidates=3600.0)
    mgr.enroll_student("Heidi", [_solid_image(140)])
    time.sleep(0.2)

    mgr.resolve(tracker_id=1, face_bgr=_solid_image(140), face_box=(0, 0, 64, 64))
    time.sleep(0.2)
    mgr.resolve(tracker_id=2, face_bgr=_solid_image(140), face_box=(0, 0, 64, 64))  # different tracker -- bypasses cache
    time.sleep(0.2)

    assert len(repo.list_pending_candidates()) == 1  # second attempt rate-limited, not a second candidate


def test_auto_update_disabled_by_default_queues_nothing(repo):
    embedder = FakeEmbedder()
    mgr = IdentityManager(
        embedder=embedder,
        quality_assessor=FakeQualityAssessor(always_pass=True),
        vector_index=LinearVectorIndex(dim=embedder.dim),
        cache=ActiveIdentityCache(),
        repository=repo,
        # auto_update_enabled defaults to False
    )
    mgr.enroll_student("Ivan", [_solid_image(100)])
    time.sleep(0.2)
    mgr.resolve(tracker_id=1, face_bgr=_solid_image(100), face_box=(0, 0, 64, 64))
    time.sleep(0.2)
    assert repo.count_pending_candidates() == 0


def test_approve_candidate_adds_embedding_and_clears_pending_row(repo):
    mgr = _manager_with_auto_update(repo)
    student, _, _ = mgr.enroll_student("Judy", [_solid_image(180)])
    time.sleep(0.2)
    mgr.resolve(tracker_id=1, face_bgr=_solid_image(180), face_box=(0, 0, 64, 64))
    time.sleep(0.2)

    candidates = repo.list_pending_candidates()
    assert len(candidates) == 1
    candidate_id = candidates[0].candidate_id

    resolved = repo.get_pending_candidate_for_approval(candidate_id)
    assert resolved is not None
    approved_student_id, vector, quality = resolved
    assert approved_student_id == student.student_id

    from core.models import FaceEmbedding
    repo.insert_embedding(FaceEmbedding(student_id=approved_student_id, vector=vector.tolist(), quality=quality, source="auto_update"))
    repo.delete_pending_candidate(candidate_id)
    time.sleep(0.2)

    assert repo.count_pending_candidates() == 0
    embeddings = repo.get_all_embeddings()
    assert len(embeddings) == 2  # original enrollment + the approved auto-update
    assert any(e.source == "auto_update" for e in embeddings)


def test_reject_candidate_leaves_no_trace(repo):
    mgr = _manager_with_auto_update(repo)
    mgr.enroll_student("Karl", [_solid_image(120)])
    time.sleep(0.2)
    mgr.resolve(tracker_id=1, face_bgr=_solid_image(120), face_box=(0, 0, 64, 64))
    time.sleep(0.2)

    candidate_id = repo.list_pending_candidates()[0].candidate_id
    repo.delete_pending_candidate(candidate_id)
    time.sleep(0.2)

    assert repo.count_pending_candidates() == 0
    assert len(repo.get_all_embeddings()) == 1  # only the original enrollment -- nothing added


def test_delete_stale_pending_candidates(repo):
    mgr = _manager_with_auto_update(repo)
    mgr.enroll_student("Liam", [_solid_image(90)])
    time.sleep(0.2)
    mgr.resolve(tracker_id=1, face_bgr=_solid_image(90), face_box=(0, 0, 64, 64))
    time.sleep(0.2)
    assert repo.count_pending_candidates() == 1

    # Cutoff in the future -- the candidate (captured "now") is older than it, so it's stale.
    repo.delete_stale_pending_candidates(datetime.now(timezone.utc) + timedelta(seconds=1))
    time.sleep(0.2)
    assert repo.count_pending_candidates() == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
