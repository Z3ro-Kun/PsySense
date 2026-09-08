"""
db/repository/repository.py

Repository layer sitting on top of AsyncDBWriter (writes) and a direct
read connection (reads). This is the only place raw SQL should appear
outside db/writer.py and db/schema.sql — every service (IdentityManager,
FusionEngine, bridge_api) talks to the database through typed methods
here, not ad-hoc cursor.execute() calls scattered through the codebase.
"""
from __future__ import annotations

import sqlite3
from typing import Optional
from uuid import UUID

import numpy as np

from core.models import FaceEmbedding, FaceQualityReport, Student
from db.writer import AsyncDBWriter, dumps, now_iso, open_read_connection


class Repository:
    def __init__(self, writer: AsyncDBWriter, db_path: str):
        self._writer = writer
        self._read_conn: sqlite3.Connection = open_read_connection(db_path)

    def close(self) -> None:
        self._read_conn.close()

    # ------------------------------------------------------------------ #
    # Students
    # ------------------------------------------------------------------ #

    def upsert_student(self, student: Student) -> None:
        self._writer.enqueue(
            """
            INSERT INTO students (student_id, display_name, external_ref, enrolled_at, active)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(student_id) DO UPDATE SET
                display_name=excluded.display_name,
                external_ref=excluded.external_ref,
                active=excluded.active
            """,
            (
                str(student.student_id),
                student.display_name,
                student.external_ref,
                student.enrolled_at.isoformat(),
                int(student.active),
            ),
        )

    def get_all_students(self) -> list[Student]:
        rows = self._read_conn.execute(
            "SELECT student_id, display_name, external_ref, enrolled_at, active FROM students WHERE active=1"
        ).fetchall()
        return [
            Student(
                student_id=UUID(r["student_id"]),
                display_name=r["display_name"],
                external_ref=r["external_ref"],
                enrolled_at=r["enrolled_at"],
                active=bool(r["active"]),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------ #
    # Face embeddings
    # ------------------------------------------------------------------ #

    def insert_embedding(self, embedding: FaceEmbedding) -> None:
        vec = np.asarray(embedding.vector, dtype=np.float32)
        q = embedding.quality
        self._writer.enqueue(
            """
            INSERT INTO face_embeddings
                (embedding_id, student_id, vector, dim, blur_score, brightness,
                 yaw_deg, occlusion_score, face_size_px, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(embedding.embedding_id),
                str(embedding.student_id),
                vec.tobytes(),
                vec.shape[0],
                q.blur_score,
                q.brightness,
                q.yaw_deg,
                q.occlusion_score,
                q.face_size_px,
                embedding.source,
                embedding.created_at.isoformat(),
            ),
        )

    def get_all_embeddings(self) -> list[FaceEmbedding]:
        """Used to rebuild the in-memory vector index on startup."""
        rows = self._read_conn.execute(
            """
            SELECT embedding_id, student_id, vector, dim, blur_score, brightness,
                   yaw_deg, occlusion_score, face_size_px, source, created_at
            FROM face_embeddings
            """
        ).fetchall()
        out = []
        for r in rows:
            vec = np.frombuffer(r["vector"], dtype=np.float32)
            quality = FaceQualityReport(
                passed=True,
                blur_score=r["blur_score"] or 0.0,
                brightness=r["brightness"] or 0.0,
                yaw_deg=r["yaw_deg"],
                occlusion_score=r["occlusion_score"] or 0.0,
                face_size_px=r["face_size_px"] or 0,
                reasons=[],
            )
            out.append(
                FaceEmbedding(
                    embedding_id=UUID(r["embedding_id"]),
                    student_id=UUID(r["student_id"]),
                    vector=vec.tolist(),
                    quality=quality,
                    source=r["source"],
                    created_at=r["created_at"],
                )
            )
        return out

    def delete_embeddings_for_student(self, student_id: UUID) -> None:
        self._writer.enqueue("DELETE FROM face_embeddings WHERE student_id = ?", (str(student_id),))

    # ------------------------------------------------------------------ #
    # Tracking sessions
    # ------------------------------------------------------------------ #

    def open_session(self, session_id: str, tracker_id: int, student_id: Optional[UUID], camera_id: str = "default") -> None:
        self._writer.enqueue(
            """
            INSERT INTO tracking_sessions (session_id, student_id, tracker_id, camera_id, start_time, status)
            VALUES (?, ?, ?, ?, ?, 'active')
            """,
            (session_id, str(student_id) if student_id else None, tracker_id, camera_id, now_iso()),
        )

    def close_session(self, session_id: str, status: str = "closed") -> None:
        self._writer.enqueue(
            "UPDATE tracking_sessions SET end_time = ?, status = ? WHERE session_id = ?",
            (now_iso(), status, session_id),
        )

    # ------------------------------------------------------------------ #
    # Events
    # ------------------------------------------------------------------ #

    def insert_emotion_event(self, session_id: Optional[str], student_id: Optional[UUID], emotions: dict[str, float]) -> None:
        dominant = max(emotions, key=emotions.get) if emotions else None
        self._writer.enqueue(
            """
            INSERT INTO emotion_events (session_id, student_id, timestamp, emotions_json, dominant_emotion)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, str(student_id) if student_id else None, now_iso(), dumps(emotions), dominant),
        )

    def insert_pose_event(
        self, session_id: Optional[str], student_id: Optional[UUID],
        slumped: float, rigidity: float, fidgeting: float,
    ) -> None:
        self._writer.enqueue(
            """
            INSERT INTO pose_events (session_id, student_id, timestamp, slumped_score, rigidity_score, fidgeting_score)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, str(student_id) if student_id else None, now_iso(), slumped, rigidity, fidgeting),
        )

    def insert_behavior_event(
        self, session_id: Optional[str], student_id: Optional[UUID],
        event_type: str, payload: dict, severity: str = "info",
    ) -> None:
        self._writer.enqueue(
            """
            INSERT INTO behavior_events (session_id, student_id, timestamp, event_type, payload_json, severity)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, str(student_id) if student_id else None, now_iso(), event_type, dumps(payload), severity),
        )

    def insert_system_log(self, level: str, component: str, message: str, context: Optional[dict] = None) -> None:
        self._writer.enqueue(
            "INSERT INTO system_logs (timestamp, level, component, message, context_json) VALUES (?, ?, ?, ?, ?)",
            (now_iso(), level, component, message, dumps(context or {})),
        )

    # ------------------------------------------------------------------ #
    # Reports
    # ------------------------------------------------------------------ #

    def upsert_daily_report(self, student_id: UUID, report_date: str, summary: dict) -> None:
        self._writer.enqueue(
            """
            INSERT INTO daily_reports (student_id, report_date, summary_json, generated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(student_id, report_date) DO UPDATE SET
                summary_json=excluded.summary_json, generated_at=excluded.generated_at
            """,
            (str(student_id), report_date, dumps(summary), now_iso()),
        )