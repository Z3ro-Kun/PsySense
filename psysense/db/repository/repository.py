"""
db/repository/repository.py

Repository layer sitting on top of AsyncDBWriter (writes) and a direct
read connection (reads). This is the only place raw SQL should appear
outside db/writer.py and db/schema.sql — every service (IdentityManager,
FusionEngine, bridge_api) talks to the database through typed methods
here, not ad-hoc cursor.execute() calls scattered through the codebase.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

import numpy as np

from core.models import (
    BehaviorEventRecord,
    EmotionEvent,
    FaceEmbedding,
    FaceQualityReport,
    PendingIdentityCandidate,
    PoseEventRecord,
    Student,
    TrackingSession,
    User,
    UserRole,
)
from db.writer import AsyncDBWriter, dumps, now_iso, open_read_connection


class Repository:
    def __init__(self, writer: AsyncDBWriter, db_path: str):
        self._writer = writer
        self._read_conn: sqlite3.Connection = open_read_connection(db_path)

    def close(self) -> None:
        self._read_conn.close()

    def flush(self, timeout: float = 2.0) -> None:
        """Blocks until all writes enqueued so far have landed. See
        AsyncDBWriter.barrier() -- use after a request-scoped batch of
        writes (e.g. enrollment) where the caller needs its own
        subsequent read to see them, not on the per-frame hot path."""
        self._writer.barrier(timeout=timeout)

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
        return [self._row_to_student(r) for r in rows]

    def get_student(self, student_id: UUID) -> Optional[Student]:
        row = self._read_conn.execute(
            "SELECT student_id, display_name, external_ref, enrolled_at, active FROM students WHERE student_id = ?",
            (str(student_id),),
        ).fetchone()
        return self._row_to_student(row) if row else None

    def delete_student(self, student_id: UUID) -> None:
        """Hard delete, not a soft-delete flag flip: schema.sql's FKs are
        explicitly designed for this -- ON DELETE CASCADE removes
        face_embeddings (biometric data shouldn't linger for a removed
        student, per the privacy guidance in CLAUDE.md Sec 19), while
        ON DELETE SET NULL on tracking_sessions/emotion_events/pose_events/
        behavior_events preserves historical analytics rows without a
        dangling student reference. Requires PRAGMA foreign_keys=ON on the
        writer's connection, which db/writer.py already sets."""
        self._writer.enqueue("DELETE FROM students WHERE student_id = ?", (str(student_id),))

    def list_students(self, limit: int = 50, offset: int = 0, search: Optional[str] = None) -> list[Student]:
        clauses = ["active = 1"]
        params: list = []
        if search:
            clauses.append("display_name LIKE ?")
            params.append(f"%{search}%")
        where = " AND ".join(clauses)
        params.extend([limit, offset])
        rows = self._read_conn.execute(
            f"""
            SELECT student_id, display_name, external_ref, enrolled_at, active
            FROM students WHERE {where}
            ORDER BY display_name ASC LIMIT ? OFFSET ?
            """,
            params,
        ).fetchall()
        return [self._row_to_student(r) for r in rows]

    def count_students(self, search: Optional[str] = None) -> int:
        clauses = ["active = 1"]
        params: list = []
        if search:
            clauses.append("display_name LIKE ?")
            params.append(f"%{search}%")
        where = " AND ".join(clauses)
        row = self._read_conn.execute(f"SELECT COUNT(*) AS n FROM students WHERE {where}", params).fetchone()
        return int(row["n"])

    @staticmethod
    def _row_to_student(r: sqlite3.Row) -> Student:
        return Student(
            student_id=UUID(r["student_id"]),
            display_name=r["display_name"],
            external_ref=r["external_ref"],
            enrolled_at=r["enrolled_at"],
            active=bool(r["active"]),
        )

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
    # Pending identity candidates (human-in-the-loop auto-update)
    # ------------------------------------------------------------------ #

    def insert_pending_candidate(
        self,
        student_id: UUID,
        vector: np.ndarray,
        image_jpeg: bytes,
        quality: FaceQualityReport,
        distance: float,
        tracker_id: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> UUID:
        candidate_id = uuid4()
        vec = np.asarray(vector, dtype=np.float32)
        self._writer.enqueue(
            """
            INSERT INTO pending_identity_candidates
                (candidate_id, student_id, tracker_id, session_id, image_jpeg, vector, dim,
                 distance, blur_score, brightness, yaw_deg, occlusion_score, face_size_px, captured_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(candidate_id), str(student_id), tracker_id, session_id, image_jpeg,
                vec.tobytes(), vec.shape[0], distance,
                quality.blur_score, quality.brightness, quality.yaw_deg, quality.occlusion_score,
                quality.face_size_px, now_iso(),
            ),
        )
        return candidate_id

    def list_pending_candidates(self, limit: int = 50, offset: int = 0) -> list[PendingIdentityCandidate]:
        rows = self._read_conn.execute(
            """
            SELECT candidate_id, student_id, tracker_id, session_id, distance,
                   blur_score, brightness, yaw_deg, occlusion_score, face_size_px, captured_at
            FROM pending_identity_candidates
            ORDER BY captured_at ASC LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
        return [self._row_to_pending_candidate(r) for r in rows]

    def count_pending_candidates(self) -> int:
        row = self._read_conn.execute("SELECT COUNT(*) AS n FROM pending_identity_candidates").fetchone()
        return int(row["n"])

    def get_pending_candidate_image(self, candidate_id: UUID) -> Optional[bytes]:
        row = self._read_conn.execute(
            "SELECT image_jpeg FROM pending_identity_candidates WHERE candidate_id = ?", (str(candidate_id),),
        ).fetchone()
        return bytes(row["image_jpeg"]) if row else None

    def get_pending_candidate_for_approval(self, candidate_id: UUID) -> Optional[tuple[UUID, np.ndarray, FaceQualityReport]]:
        """Returns (student_id, vector, quality) needed to build a real
        FaceEmbedding on approval, or None if the candidate no longer
        exists (already resolved, or expired)."""
        row = self._read_conn.execute(
            """
            SELECT student_id, vector, blur_score, brightness, yaw_deg, occlusion_score, face_size_px
            FROM pending_identity_candidates WHERE candidate_id = ?
            """,
            (str(candidate_id),),
        ).fetchone()
        if row is None:
            return None
        vec = np.frombuffer(row["vector"], dtype=np.float32)
        quality = FaceQualityReport(
            passed=True,
            blur_score=row["blur_score"] or 0.0,
            brightness=row["brightness"] or 0.0,
            yaw_deg=row["yaw_deg"],
            occlusion_score=row["occlusion_score"] or 0.0,
            face_size_px=row["face_size_px"] or 0,
            reasons=[],
        )
        return UUID(row["student_id"]), vec, quality

    def delete_pending_candidate(self, candidate_id: UUID) -> None:
        self._writer.enqueue("DELETE FROM pending_identity_candidates WHERE candidate_id = ?", (str(candidate_id),))

    def delete_stale_pending_candidates(self, older_than: datetime) -> None:
        """Unreviewed candidates past the retention window are discarded
        unconditionally -- raw face images never accumulate indefinitely
        just because nobody reviewed them in time."""
        self._writer.enqueue(
            "DELETE FROM pending_identity_candidates WHERE captured_at < ?",
            (older_than.strftime("%Y-%m-%dT%H:%M:%S"),),
        )

    def count_recent_pending_candidates_for_student(self, student_id: UUID, since: datetime) -> int:
        """Used to rate-limit candidate creation (identity.auto_update.
        min_seconds_between_candidates_per_student) -- a DB round-trip
        rather than in-memory state so the rate limit holds even across a
        pipeline restart."""
        row = self._read_conn.execute(
            "SELECT COUNT(*) AS n FROM pending_identity_candidates WHERE student_id = ? AND captured_at >= ?",
            (str(student_id), since.strftime("%Y-%m-%dT%H:%M:%S")),
        ).fetchone()
        return int(row["n"])

    @staticmethod
    def _row_to_pending_candidate(r: sqlite3.Row) -> PendingIdentityCandidate:
        quality = FaceQualityReport(
            passed=True,
            blur_score=r["blur_score"] or 0.0,
            brightness=r["brightness"] or 0.0,
            yaw_deg=r["yaw_deg"],
            occlusion_score=r["occlusion_score"] or 0.0,
            face_size_px=r["face_size_px"] or 0,
            reasons=[],
        )
        return PendingIdentityCandidate(
            candidate_id=UUID(r["candidate_id"]),
            student_id=UUID(r["student_id"]),
            tracker_id=r["tracker_id"],
            session_id=r["session_id"],
            distance=r["distance"],
            quality=quality,
            captured_at=r["captured_at"],
        )

    # ------------------------------------------------------------------ #
    # Users (sub-admin accounts) and reviewer assignments
    # ------------------------------------------------------------------ #

    def create_user(self, username: str, password_hash: str, role: UserRole = UserRole.REVIEWER) -> UUID:
        user_id = uuid4()
        self._writer.enqueue(
            "INSERT INTO users (user_id, username, password_hash, role, active, created_at) VALUES (?, ?, ?, ?, 1, ?)",
            (str(user_id), username, password_hash, role.value, now_iso()),
        )
        return user_id

    def get_user_by_username(self, username: str) -> Optional[tuple[User, str]]:
        """Returns (User, password_hash) -- the hash is only ever handed
        to api/auth.py's verify_password(), never serialized into an API
        response (see User's docstring)."""
        row = self._read_conn.execute(
            "SELECT user_id, username, password_hash, role, active, created_at FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_user(row), row["password_hash"]

    def get_user(self, user_id: UUID) -> Optional[User]:
        row = self._read_conn.execute(
            "SELECT user_id, username, password_hash, role, active, created_at FROM users WHERE user_id = ?",
            (str(user_id),),
        ).fetchone()
        return self._row_to_user(row) if row else None

    def list_users(self) -> list[User]:
        rows = self._read_conn.execute(
            "SELECT user_id, username, password_hash, role, active, created_at FROM users ORDER BY created_at ASC"
        ).fetchall()
        return [self._row_to_user(r) for r in rows]

    def set_user_active(self, user_id: UUID, active: bool) -> None:
        self._writer.enqueue("UPDATE users SET active = ? WHERE user_id = ?", (int(active), str(user_id)))

    @staticmethod
    def _row_to_user(r: sqlite3.Row) -> User:
        return User(
            user_id=UUID(r["user_id"]), username=r["username"], role=UserRole(r["role"]),
            active=bool(r["active"]), created_at=r["created_at"],
        )

    def set_reviewer_assignments(self, user_id: UUID, student_ids: list[UUID]) -> None:
        """Replace semantics: the reviewer's assigned-students set becomes
        exactly student_ids. Two enqueued jobs (delete-all, then
        insert-each) execute strictly in order on AsyncDBWriter's single
        writer thread, so this is safe without an explicit transaction."""
        self._writer.enqueue("DELETE FROM reviewer_assignments WHERE user_id = ?", (str(user_id),))
        assigned_at = now_iso()
        for student_id in student_ids:
            self._writer.enqueue(
                "INSERT INTO reviewer_assignments (user_id, student_id, assigned_at) VALUES (?, ?, ?)",
                (str(user_id), str(student_id), assigned_at),
            )

    def get_reviewer_assignments(self, user_id: UUID) -> list[UUID]:
        rows = self._read_conn.execute(
            "SELECT student_id FROM reviewer_assignments WHERE user_id = ?", (str(user_id),),
        ).fetchall()
        return [UUID(r["student_id"]) for r in rows]

    def count_reviewer_assignments(self, user_id: UUID) -> int:
        row = self._read_conn.execute(
            "SELECT COUNT(*) AS n FROM reviewer_assignments WHERE user_id = ?", (str(user_id),),
        ).fetchone()
        return int(row["n"])

    def is_student_assigned_to_reviewer(self, user_id: UUID, student_id: UUID) -> bool:
        row = self._read_conn.execute(
            "SELECT 1 FROM reviewer_assignments WHERE user_id = ? AND student_id = ?",
            (str(user_id), str(student_id)),
        ).fetchone()
        return row is not None

    def list_pending_candidates_for_reviewer(self, user_id: UUID, limit: int = 50, offset: int = 0) -> list[PendingIdentityCandidate]:
        rows = self._read_conn.execute(
            """
            SELECT c.candidate_id, c.student_id, c.tracker_id, c.session_id, c.distance,
                   c.blur_score, c.brightness, c.yaw_deg, c.occlusion_score, c.face_size_px, c.captured_at
            FROM pending_identity_candidates c
            JOIN reviewer_assignments a ON a.student_id = c.student_id AND a.user_id = ?
            ORDER BY c.captured_at ASC LIMIT ? OFFSET ?
            """,
            (str(user_id), limit, offset),
        ).fetchall()
        return [self._row_to_pending_candidate(r) for r in rows]

    def count_pending_candidates_for_reviewer(self, user_id: UUID) -> int:
        row = self._read_conn.execute(
            """
            SELECT COUNT(*) AS n FROM pending_identity_candidates c
            JOIN reviewer_assignments a ON a.student_id = c.student_id AND a.user_id = ?
            """,
            (str(user_id),),
        ).fetchone()
        return int(row["n"])

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

    # ------------------------------------------------------------------ #
    # Paginated reads for the API layer
    # ------------------------------------------------------------------ #

    @staticmethod
    def _time_range_filter(
        student_id: Optional[UUID], since: Optional[datetime], until: Optional[datetime], time_col: str = "timestamp"
    ) -> tuple[list[str], list]:
        clauses: list[str] = []
        params: list = []
        if student_id is not None:
            clauses.append("student_id = ?")
            params.append(str(student_id))
        if since is not None:
            clauses.append(f"{time_col} >= ?")
            params.append(since.strftime("%Y-%m-%dT%H:%M:%S"))
        if until is not None:
            clauses.append(f"{time_col} <= ?")
            params.append(until.strftime("%Y-%m-%dT%H:%M:%S"))
        return clauses, params

    def list_sessions(
        self, limit: int = 50, offset: int = 0,
        student_id: Optional[UUID] = None, since: Optional[datetime] = None, until: Optional[datetime] = None,
    ) -> list[TrackingSession]:
        clauses, params = self._time_range_filter(student_id, since, until, time_col="start_time")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._read_conn.execute(
            f"""
            SELECT session_id, student_id, tracker_id, camera_id, start_time, end_time, status
            FROM tracking_sessions {where}
            ORDER BY start_time DESC LIMIT ? OFFSET ?
            """,
            (*params, limit, offset),
        ).fetchall()
        return [
            TrackingSession(
                session_id=r["session_id"],
                student_id=UUID(r["student_id"]) if r["student_id"] else None,
                tracker_id=r["tracker_id"],
                camera_id=r["camera_id"],
                start_time=r["start_time"],
                end_time=r["end_time"],
                status=r["status"],
            )
            for r in rows
        ]

    def count_sessions(
        self, student_id: Optional[UUID] = None, since: Optional[datetime] = None, until: Optional[datetime] = None,
    ) -> int:
        clauses, params = self._time_range_filter(student_id, since, until, time_col="start_time")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        row = self._read_conn.execute(f"SELECT COUNT(*) AS n FROM tracking_sessions {where}", params).fetchone()
        return int(row["n"])

    def list_emotion_events(
        self, limit: int = 100, offset: int = 0,
        student_id: Optional[UUID] = None, since: Optional[datetime] = None, until: Optional[datetime] = None,
    ) -> list[EmotionEvent]:
        clauses, params = self._time_range_filter(student_id, since, until)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._read_conn.execute(
            f"""
            SELECT event_id, session_id, student_id, timestamp, emotions_json, dominant_emotion
            FROM emotion_events {where}
            ORDER BY timestamp DESC LIMIT ? OFFSET ?
            """,
            (*params, limit, offset),
        ).fetchall()
        return [
            EmotionEvent(
                event_id=r["event_id"],
                session_id=r["session_id"],
                student_id=UUID(r["student_id"]) if r["student_id"] else None,
                timestamp=r["timestamp"],
                emotions=json.loads(r["emotions_json"]),
                dominant_emotion=r["dominant_emotion"],
            )
            for r in rows
        ]

    def count_emotion_events(
        self, student_id: Optional[UUID] = None, since: Optional[datetime] = None, until: Optional[datetime] = None,
    ) -> int:
        clauses, params = self._time_range_filter(student_id, since, until)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        row = self._read_conn.execute(f"SELECT COUNT(*) AS n FROM emotion_events {where}", params).fetchone()
        return int(row["n"])

    def list_pose_events(
        self, limit: int = 100, offset: int = 0,
        student_id: Optional[UUID] = None, since: Optional[datetime] = None, until: Optional[datetime] = None,
    ) -> list[PoseEventRecord]:
        clauses, params = self._time_range_filter(student_id, since, until)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._read_conn.execute(
            f"""
            SELECT event_id, session_id, student_id, timestamp, slumped_score, rigidity_score, fidgeting_score
            FROM pose_events {where}
            ORDER BY timestamp DESC LIMIT ? OFFSET ?
            """,
            (*params, limit, offset),
        ).fetchall()
        return [
            PoseEventRecord(
                event_id=r["event_id"],
                session_id=r["session_id"],
                student_id=UUID(r["student_id"]) if r["student_id"] else None,
                timestamp=r["timestamp"],
                slumped_score=r["slumped_score"],
                rigidity_score=r["rigidity_score"],
                fidgeting_score=r["fidgeting_score"],
            )
            for r in rows
        ]

    def count_pose_events(
        self, student_id: Optional[UUID] = None, since: Optional[datetime] = None, until: Optional[datetime] = None,
    ) -> int:
        clauses, params = self._time_range_filter(student_id, since, until)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        row = self._read_conn.execute(f"SELECT COUNT(*) AS n FROM pose_events {where}", params).fetchone()
        return int(row["n"])

    def list_behavior_events(
        self, limit: int = 100, offset: int = 0,
        student_id: Optional[UUID] = None, since: Optional[datetime] = None, until: Optional[datetime] = None,
        severity: Optional[str] = None,
    ) -> list[BehaviorEventRecord]:
        clauses, params = self._time_range_filter(student_id, since, until)
        if severity is not None:
            clauses.append("severity = ?")
            params.append(severity)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._read_conn.execute(
            f"""
            SELECT event_id, session_id, student_id, timestamp, event_type, payload_json, severity
            FROM behavior_events {where}
            ORDER BY timestamp DESC LIMIT ? OFFSET ?
            """,
            (*params, limit, offset),
        ).fetchall()
        return [
            BehaviorEventRecord(
                event_id=r["event_id"],
                session_id=r["session_id"],
                student_id=UUID(r["student_id"]) if r["student_id"] else None,
                timestamp=r["timestamp"],
                event_type=r["event_type"],
                payload=json.loads(r["payload_json"]) if r["payload_json"] else {},
                severity=r["severity"],
            )
            for r in rows
        ]

    def count_behavior_events(
        self, student_id: Optional[UUID] = None, since: Optional[datetime] = None, until: Optional[datetime] = None,
        severity: Optional[str] = None,
    ) -> int:
        clauses, params = self._time_range_filter(student_id, since, until)
        if severity is not None:
            clauses.append("severity = ?")
            params.append(severity)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        row = self._read_conn.execute(f"SELECT COUNT(*) AS n FROM behavior_events {where}", params).fetchone()
        return int(row["n"])

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