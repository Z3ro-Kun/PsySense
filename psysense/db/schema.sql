-- db/schema.sql
-- Normalized PsySense v2 schema.
--
-- Design notes:
--   * Identity is the anchor. student_id (a UUID string) is the foreign
--     key threaded through every event table -- tracker_id only appears
--     in tracking_sessions, as a session-scoped attribute, never as the
--     join key for anything else. This is the DB-level enforcement of
--     "decouple identity from tracker IDs".
--   * face_embeddings stores VECTOR HISTORY (one row per accepted
--     enrollment image), never overwritten -- matches the requirement
--     to support multiple enrollment images per student.
--   * Vectors are stored as raw float32 BLOBs (via numpy .tobytes()),
--     not JSON -- compact and avoids float-string round-trip issues.
--     The FAISS/linear index is the actual query-time source of truth;
--     these rows are the durable copy used to rebuild the index on
--     startup or after an index-backend switch.
--   * Foreign keys use ON DELETE CASCADE for face_embeddings (embeddings
--     are meaningless without their student) but ON DELETE SET NULL for
--     event tables (so historical emotion/pose data survives a student
--     record being deactivated/merged, rather than silently vanishing).
--   * WAL journal mode (set by db/writer.py at connection time, not
--     here) lets readers (bridge_api) run concurrently with the single
--     writer without lock contention.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS students (
    student_id      TEXT PRIMARY KEY,
    display_name    TEXT NOT NULL,
    external_ref    TEXT,
    enrolled_at     TEXT NOT NULL,
    active          INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    embedding_id    TEXT PRIMARY KEY,
    student_id      TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    vector          BLOB NOT NULL,
    dim             INTEGER NOT NULL,
    blur_score      REAL,
    brightness      REAL,
    yaw_deg         REAL,
    occlusion_score REAL,
    face_size_px    INTEGER,
    source          TEXT NOT NULL DEFAULT 'enrollment',
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_student ON face_embeddings(student_id);

CREATE TABLE IF NOT EXISTS tracking_sessions (
    session_id      TEXT PRIMARY KEY,
    student_id      TEXT REFERENCES students(student_id) ON DELETE SET NULL,
    tracker_id      INTEGER NOT NULL,
    camera_id       TEXT NOT NULL DEFAULT 'default',
    start_time      TEXT NOT NULL,
    end_time        TEXT,
    status          TEXT NOT NULL DEFAULT 'active'   -- active | closed | lost
);
CREATE INDEX IF NOT EXISTS idx_sessions_student ON tracking_sessions(student_id);
CREATE INDEX IF NOT EXISTS idx_sessions_tracker ON tracking_sessions(tracker_id);

CREATE TABLE IF NOT EXISTS emotion_events (
    event_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT REFERENCES tracking_sessions(session_id) ON DELETE SET NULL,
    student_id        TEXT REFERENCES students(student_id) ON DELETE SET NULL,
    timestamp         TEXT NOT NULL,
    emotions_json     TEXT NOT NULL,      -- {"happy":0.2,"neutral":0.6,...}
    dominant_emotion  TEXT
);
CREATE INDEX IF NOT EXISTS idx_emotion_student_time ON emotion_events(student_id, timestamp);

CREATE TABLE IF NOT EXISTS pose_events (
    event_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT REFERENCES tracking_sessions(session_id) ON DELETE SET NULL,
    student_id        TEXT REFERENCES students(student_id) ON DELETE SET NULL,
    timestamp         TEXT NOT NULL,
    slumped_score     REAL NOT NULL DEFAULT 0.0,
    rigidity_score    REAL NOT NULL DEFAULT 0.0,
    fidgeting_score   REAL NOT NULL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_pose_student_time ON pose_events(student_id, timestamp);

CREATE TABLE IF NOT EXISTS behavior_events (
    event_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT REFERENCES tracking_sessions(session_id) ON DELETE SET NULL,
    student_id        TEXT REFERENCES students(student_id) ON DELETE SET NULL,
    timestamp         TEXT NOT NULL,
    event_type        TEXT NOT NULL,       -- e.g. "prolonged_slump", "disengagement"
    payload_json      TEXT,                -- fusion-engine output that triggered this event
    severity          TEXT NOT NULL DEFAULT 'info'   -- info | warning | critical
);
CREATE INDEX IF NOT EXISTS idx_behavior_student_time ON behavior_events(student_id, timestamp);

CREATE TABLE IF NOT EXISTS daily_reports (
    report_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id      TEXT REFERENCES students(student_id) ON DELETE CASCADE,
    report_date     TEXT NOT NULL,          -- YYYY-MM-DD
    summary_json    TEXT NOT NULL,
    generated_at    TEXT NOT NULL,
    UNIQUE(student_id, report_date)
);

CREATE TABLE IF NOT EXISTS system_logs (
    log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    level           TEXT NOT NULL,          -- DEBUG|INFO|WARNING|ERROR|CRITICAL
    component       TEXT NOT NULL,          -- e.g. "identity_manager", "db_writer"
    message         TEXT NOT NULL,
    context_json    TEXT
);
CREATE INDEX IF NOT EXISTS idx_logs_time ON system_logs(timestamp);