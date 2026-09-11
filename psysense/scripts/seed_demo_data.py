"""
scripts/seed_demo_data.py

Populates the database with realistic-looking fake students and
emotion/pose/behavior history -- for a portfolio/demo deployment (no
camera, no real ML models) or for generating dashboard screenshots.
Never touches real biometric data: face embeddings are random vectors,
not derived from any actual photo, and are only ever added via the
existing IdentityManager.enroll_student() path (so quality-gate/dedup
logic is exercised the same way real enrollment would use it).

Run from psysense/ (uses config.yaml's database path unless --db-path is given):
    python -m scripts.seed_demo_data
    python -m scripts.seed_demo_data --db-path data/demo.db --reset
"""
from __future__ import annotations

import argparse
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.config import load_config
from db.repository.repository import Repository
from db.writer import AsyncDBWriter
from services.identity.cache import ActiveIdentityCache
from services.identity.manager import IdentityManager
from services.identity.vector_index import LinearVectorIndex

DEMO_NAMES = [
    ("Ava Martinez", "S-1001"), ("Liam Chen", "S-1002"), ("Sofia Rossi", "S-1003"),
    ("Noah Kim", "S-1004"), ("Maya Patel", "S-1005"), ("Ethan Brooks", "S-1006"),
]
EMOTIONS = ("happy", "sad", "angry", "fear", "disgust", "surprise", "neutral")


class _DemoEmbedder:
    """Deterministic fake embedder -- same contract as
    tests/test_identity_manager.py's FakeEmbedder, so enroll_student()'s
    real quality-gate/dedup/persistence logic runs unchanged. Not a
    stand-in for real face recognition; demo data is for UI screenshots
    and a seeded cloud deployment, not identity testing."""

    dim = 16

    def analyze(self, face_bgr):
        return None

    def embed(self, face_bgr):
        import numpy as np
        rng = random.Random(id(face_bgr))
        vec = np.array([rng.uniform(-1, 1) for _ in range(self.dim)], dtype=np.float32)
        norm = float(np.linalg.norm(vec)) or 1.0
        return vec / norm


class _DemoQuality:
    def assess(self, face_bgr, face_box, landmarks_5pt=None):
        from core.models import FaceQualityReport
        return FaceQualityReport(passed=True, blur_score=200.0, brightness=130.0, occlusion_score=0.0, face_size_px=120, reasons=[])


def _fake_image(seed: int):
    import numpy as np
    return np.full((64, 64, 3), seed % 255, dtype="uint8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the database with demo students and behavioral history")
    parser.add_argument("--db-path", default=None, help="Override config.yaml's database path")
    parser.add_argument("--days", type=int, default=5, help="How many days of history to generate per student")
    parser.add_argument("--events-per-day", type=int, default=6, help="Emotion/pose windows per student per day")
    args = parser.parse_args()

    config = load_config()
    db_path = args.db_path or config.pipeline.database.path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    schema_path = str(Path(__file__).resolve().parents[1] / "db" / "schema.sql")

    writer = AsyncDBWriter(db_path=db_path, schema_path=schema_path)
    writer.start()
    time.sleep(0.2)
    repo = Repository(writer=writer, db_path=db_path)

    identity = IdentityManager(
        embedder=_DemoEmbedder(), quality_assessor=_DemoQuality(),
        vector_index=LinearVectorIndex(dim=_DemoEmbedder.dim), cache=ActiveIdentityCache(), repository=repo,
    )

    rng = random.Random(42)
    print(f"Seeding demo data into {db_path} ...")

    for i, (name, roll) in enumerate(DEMO_NAMES):
        student, accepted, _ = identity.enroll_student(name, [_fake_image(i * 37 + j) for j in range(2)], external_ref=roll)
        session_id = f"demo-session-{i}"
        repo.open_session(session_id, tracker_id=i + 1, student_id=student.student_id, camera_id="demo_camera")

        base_happy = rng.uniform(0.25, 0.55)
        now = datetime.now(timezone.utc)
        for day in range(args.days):
            for slot in range(args.events_per_day):
                ts = now - timedelta(days=args.days - day, hours=rng.uniform(0, 6), minutes=slot * 7)
                happy = max(0.02, min(0.8, base_happy + rng.uniform(-0.2, 0.2)))
                sad = max(0.0, rng.uniform(0.05, 0.35))
                neutral = max(0.05, 1.0 - happy - sad)
                total = happy + sad + neutral
                emotions = {"happy": happy / total, "sad": sad / total, "neutral": neutral / total}
                repo.insert_emotion_event(session_id, student.student_id, emotions, timestamp=ts)

                slumped = max(0.0, min(1.0, rng.uniform(0.1, 0.7)))
                rigidity = max(0.0, min(1.0, rng.uniform(0.1, 0.5)))
                fidgeting = max(0.0, min(1.0, rng.uniform(0.05, 0.6)))
                repo.insert_pose_event(session_id, student.student_id, slumped, rigidity, fidgeting, timestamp=ts)

                if slumped > 0.6 and rng.random() < 0.3:
                    repo.insert_behavior_event(
                        session_id, student.student_id, "prolonged_slump",
                        {"detail": f"Slumped posture sustained (score={slumped:.2f})"}, severity="warning",
                    )

        repo.close_session(session_id, status="closed")
        print(f"  {name} ({roll}): {len(accepted)} demo embedding(s), {args.days * args.events_per_day} windows")

    time.sleep(0.3)
    repo.close()
    writer.stop()
    print("Done.")


if __name__ == "__main__":
    main()
