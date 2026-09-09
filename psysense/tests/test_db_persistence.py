"""
tests/test_db_persistence.py

Integration tests covering:
  - AsyncDBWriter correctly applies schema + serializes writes
  - Repository read/write round-trips for students and embeddings
  - IdentityManager persists on enroll() and correctly rehydrates in a
    brand new process-equivalent instance (proves restart survival)
  - A genuine concurrent read-while-write check against a real SQLite
    file, to validate the WAL configuration actually avoids "database is
    locked" errors under load
"""
from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.writer import AsyncDBWriter, open_read_connection  # noqa: E402
from db.repository.repository import Repository  # noqa: E402
from services.identity.cache import ActiveIdentityCache  # noqa: E402
from services.identity.manager import IdentityManager  # noqa: E402
from services.identity.vector_index import LinearVectorIndex  # noqa: E402
from core.models import MatchStatus  # noqa: E402
from tests.test_identity_manager import FakeEmbedder, FakeQualityAssessor, _solid_image  # noqa: E402

SCHEMA_PATH = str(Path(__file__).resolve().parents[1] / "db" / "schema.sql")


@pytest.fixture()
def db_path(tmp_path):
    return str(tmp_path / "psysense_test.db")


@pytest.fixture()
def writer(db_path):
    w = AsyncDBWriter(db_path=db_path, schema_path=SCHEMA_PATH)
    w.start()
    yield w
    w.stop()


def _make_repo(writer: AsyncDBWriter, db_path: str) -> Repository:
    # Give the writer thread a beat to apply the schema before reads happen.
    time.sleep(0.1)
    return Repository(writer=writer, db_path=db_path)


def test_writer_applies_schema_and_writes_student(writer, db_path):
    repo = _make_repo(writer, db_path)
    from core.models import Student

    student = Student(display_name="Alice")
    repo.upsert_student(student)
    time.sleep(0.2)  # allow async write to land

    students = repo.get_all_students()
    assert len(students) == 1
    assert students[0].display_name == "Alice"
    assert students[0].student_id == student.student_id
    repo.close()


def test_embedding_round_trip(writer, db_path):
    repo = _make_repo(writer, db_path)
    from core.models import FaceEmbedding, FaceQualityReport, Student

    student = Student(display_name="Bob")
    repo.upsert_student(student)

    quality = FaceQualityReport(
        passed=True, blur_score=150.0, brightness=110.0, yaw_deg=2.0,
        occlusion_score=0.05, face_size_px=120, reasons=[],
    )
    vec = np.random.rand(512).astype(np.float32)
    vec /= np.linalg.norm(vec)
    embedding = FaceEmbedding(student_id=student.student_id, vector=vec.tolist(), quality=quality)
    repo.insert_embedding(embedding)
    time.sleep(0.2)

    loaded = repo.get_all_embeddings()
    assert len(loaded) == 1
    np.testing.assert_allclose(loaded[0].as_array(), vec, rtol=1e-5)
    assert loaded[0].student_id == student.student_id
    repo.close()


def test_identity_manager_survives_restart(writer, db_path):
    """Enroll via one IdentityManager instance, then build a brand-new
    instance (simulating a process restart) pointed at the same DB, and
    confirm it can still resolve the previously enrolled identity purely
    from persisted state."""
    repo = _make_repo(writer, db_path)
    embedder = FakeEmbedder()

    mgr1 = IdentityManager(
        embedder=embedder,
        quality_assessor=FakeQualityAssessor(always_pass=True),
        vector_index=LinearVectorIndex(dim=embedder.dim),
        cache=ActiveIdentityCache(),
        repository=repo,
    )
    student, accepted, _ = mgr1.enroll_student("Carol", [_solid_image(180)])
    assert len(accepted) == 1
    time.sleep(0.2)  # allow async writes to land before "restart"

    # Simulate a fresh process: new manager, new (empty) vector index,
    # same DB file.
    repo2 = Repository(writer=writer, db_path=db_path)
    mgr2 = IdentityManager(
        embedder=embedder,
        quality_assessor=FakeQualityAssessor(always_pass=True),
        vector_index=LinearVectorIndex(dim=embedder.dim),
        cache=ActiveIdentityCache(),
        repository=repo2,
    )
    mgr2.load_from_repository()

    result = mgr2.resolve(tracker_id=99, face_bgr=_solid_image(180), face_box=(0, 0, 64, 64))
    assert result.status == MatchStatus.MATCHED
    assert result.student_id == student.student_id
    repo.close()
    repo2.close()


def test_refresh_from_repository_picks_up_cross_process_enrollment(writer, db_path):
    """Simulates the bridge_api scenario: one IdentityManager (standing in
    for the running pipeline) is already resolving faces when a SEPARATE
    IdentityManager instance (standing in for the API process) enrolls a
    new student against the same DB. The pipeline's manager must pick up
    the new student via refresh_from_repository() without a restart, and
    must not re-add embeddings it already has (no duplicate index rows)."""
    repo_pipeline = _make_repo(writer, db_path)
    repo_api = Repository(writer=writer, db_path=db_path)
    embedder = FakeEmbedder()

    pipeline_mgr = IdentityManager(
        embedder=embedder,
        quality_assessor=FakeQualityAssessor(always_pass=True),
        vector_index=LinearVectorIndex(dim=embedder.dim),
        cache=ActiveIdentityCache(),
        repository=repo_pipeline,
    )
    pipeline_mgr.load_from_repository()  # empty DB at this point
    assert pipeline_mgr.resolve(tracker_id=1, face_bgr=_solid_image(90), face_box=(0, 0, 64, 64)).status == MatchStatus.UNKNOWN

    api_mgr = IdentityManager(
        embedder=embedder,
        quality_assessor=FakeQualityAssessor(always_pass=True),
        vector_index=LinearVectorIndex(dim=embedder.dim),
        repository=repo_api,
    )
    student, accepted, _ = api_mgr.enroll_student("Dana", [_solid_image(90)])
    assert len(accepted) == 1
    time.sleep(0.2)  # allow async writes to land

    added = pipeline_mgr.refresh_from_repository()
    assert added == 1

    result = pipeline_mgr.resolve(tracker_id=2, face_bgr=_solid_image(90), face_box=(0, 0, 64, 64))
    assert result.status == MatchStatus.MATCHED
    assert result.student_id == student.student_id

    # A second refresh with no new data must be a no-op, not a re-add.
    assert pipeline_mgr.refresh_from_repository() == 0

    repo_pipeline.close()
    repo_api.close()


def test_refresh_from_repository_picks_up_cross_process_deletion(writer, db_path):
    """Mirrors the enrollment test above, but for deletion: a student
    removed via a SEPARATE process (bridge_api's DELETE endpoint calling
    Repository.delete_student()) must stop being recognized by the
    already-running pipeline's IdentityManager on its next
    refresh_from_repository() cycle, not just disappear from listings."""
    repo_pipeline = _make_repo(writer, db_path)
    repo_api = Repository(writer=writer, db_path=db_path)
    embedder = FakeEmbedder()

    pipeline_mgr = IdentityManager(
        embedder=embedder,
        quality_assessor=FakeQualityAssessor(always_pass=True),
        vector_index=LinearVectorIndex(dim=embedder.dim),
        cache=ActiveIdentityCache(),
        repository=repo_pipeline,
    )
    student, accepted, _ = pipeline_mgr.enroll_student("Erin", [_solid_image(210)])
    assert len(accepted) == 1
    time.sleep(0.2)

    result = pipeline_mgr.resolve(tracker_id=1, face_bgr=_solid_image(210), face_box=(0, 0, 64, 64))
    assert result.status == MatchStatus.MATCHED

    # Deletion happens via a different process/instance, same DB.
    repo_api.delete_student(student.student_id)
    time.sleep(0.2)

    removed_before_refresh = pipeline_mgr.resolve(
        tracker_id=1, face_bgr=_solid_image(210), face_box=(0, 0, 64, 64),
    )
    # tracker_id=1 is still cache-hit at this point (TTL hasn't expired) --
    # that staleness window is expected and bounded, not what this test
    # checks. Use a fresh tracker_id to force an actual index search.
    assert removed_before_refresh.status == MatchStatus.CACHE_HIT

    pipeline_mgr.refresh_from_repository()

    result_after = pipeline_mgr.resolve(tracker_id=2, face_bgr=_solid_image(210), face_box=(0, 0, 64, 64))
    assert result_after.status == MatchStatus.UNKNOWN

    repo_pipeline.close()
    repo_api.close()


def test_concurrent_read_during_write_does_not_lock(writer, db_path):
    """Real SQLite file, real WAL mode, real thread doing writes while
    another thread reads concurrently — this is the exact failure mode
    (database is locked) the single-writer + WAL design is meant to
    eliminate."""
    repo = _make_repo(writer, db_path)
    errors: list[str] = []
    stop = threading.Event()

    def writer_loop():
        for i in range(50):
            repo.insert_system_log("INFO", "test", f"message {i}")

    def reader_loop():
        conn = open_read_connection(db_path)
        try:
            while not stop.is_set():
                try:
                    conn.execute("SELECT COUNT(*) FROM system_logs").fetchone()
                except Exception as exc:  # noqa: BLE001
                    errors.append(str(exc))
                time.sleep(0.01)
        finally:
            conn.close()

    t_reader = threading.Thread(target=reader_loop)
    t_reader.start()

    t_writer = threading.Thread(target=writer_loop)
    t_writer.start()
    t_writer.join()

    time.sleep(0.3)
    stop.set()
    t_reader.join(timeout=2.0)

    assert errors == [], f"Reader hit errors during concurrent writes: {errors}"
    repo.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))