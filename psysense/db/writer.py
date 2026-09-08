"""
db/writer.py

AsyncDBWriter: the single writer for the entire application. Every write
anywhere in PsySense — identity enrollment, emotion events, pose events,
behavior events, session lifecycle, system logs — goes through this one
queue and one background thread, which owns the one and only write
connection to the SQLite file.

Why: the original codebase had Main.py writing directly AND bridge_api.py
opening its own connection to the same file per request. SQLite allows
one writer at a time; under WAL mode readers don't block the writer and
vice versa, but two independent writers can still collide ("database is
locked"). Funneling all writes through a single serialized queue removes
that class of failure entirely, and readers (bridge_api, benchmarks) get
their own separate read-only connections against the WAL file.

This is intentionally a plain thread + queue.Queue, not asyncio — the
callers (OpenCV capture loop, FastAPI sync endpoints, benchmark scripts)
are a mix of sync and async contexts, and a plain thread-safe queue is
the simplest thing that works correctly from all of them without forcing
an event loop everywhere.
"""
from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from core.exceptions import DatabaseWriteError

logger = logging.getLogger("psysense.db.writer")


@dataclass
class _WriteJob:
    sql: str
    params: tuple
    result_event: Optional[threading.Event] = None
    result_holder: dict = field(default_factory=dict)  # {'rowid': ..., 'error': ...}


class AsyncDBWriter:
    def __init__(self, db_path: str, schema_path: Optional[str] = None, queue_maxsize: int = 10_000):
        self.db_path = db_path
        self.schema_path = schema_path
        self._queue: "queue.Queue[_WriteJob | None]" = queue.Queue(maxsize=queue_maxsize)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._dropped_jobs = 0

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self._running:
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Apply schema synchronously, in the caller's thread, before
        # spawning the writer thread or returning control. Previously
        # this only happened inside the background thread's own
        # connection setup, which meant any reader (e.g. Repository
        # constructed immediately after start()) could race ahead of
        # schema creation and hit "no such table" -- exactly what
        # happened when main.py built a Repository right after
        # writer.start() with no models/tables yet in a fresh DB file.
        self._apply_schema_sync()

        self._running = True
        self._thread = threading.Thread(target=self._run, name="psysense-db-writer", daemon=True)
        self._thread.start()
        logger.info("AsyncDBWriter started (db_path=%s)", self.db_path)

    def _apply_schema_sync(self) -> None:
        if not self.schema_path or not Path(self.schema_path).exists():
            return
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            with open(self.schema_path, "r", encoding="utf-8") as f:
                conn.executescript(f.read())
            conn.commit()
        finally:
            conn.close()

    def stop(self, timeout: float = 5.0) -> None:
        if not self._running:
            return
        self._running = False
        try:
            self._queue.put_nowait(None)  # sentinel to unblock the worker
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        logger.info("AsyncDBWriter stopped (dropped_jobs=%d)", self._dropped_jobs)

    # ------------------------------------------------------------------ #
    def enqueue(self, sql: str, params: tuple = ()) -> None:
        """Fire-and-forget write. Non-blocking; if the queue is full (writer
        can't keep up, or DB is down) the job is dropped and counted rather
        than blocking the caller's hot path (the video/inference loop)."""
        job = _WriteJob(sql=sql, params=params)
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            self._dropped_jobs += 1
            logger.warning("DB write queue full — dropping job (total dropped=%d)", self._dropped_jobs)

    def execute_returning_id(self, sql: str, params: tuple = (), timeout: float = 2.0) -> Optional[int]:
        """Blocking write that returns lastrowid — use sparingly (e.g. when
        the caller genuinely needs the new session_id back), not on the
        per-frame hot path."""
        job = _WriteJob(sql=sql, params=params, result_event=threading.Event())
        try:
            self._queue.put(job, timeout=timeout)
        except queue.Full as exc:
            raise DatabaseWriteError("DB write queue full") from exc

        if not job.result_event.wait(timeout=timeout):
            raise DatabaseWriteError("DB write timed out waiting for writer thread")
        if "error" in job.result_holder:
            raise DatabaseWriteError(job.result_holder["error"])
        return job.result_holder.get("rowid")

    @property
    def dropped_jobs(self) -> int:
        return self._dropped_jobs

    def queue_depth(self) -> int:
        return self._queue.qsize()

    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        conn = self._open_connection()
        try:
            while self._running or not self._queue.empty():
                try:
                    job = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if job is None:  # stop() sentinel
                    break
                self._execute_job(conn, job)
        finally:
            conn.close()

    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        if self.schema_path and Path(self.schema_path).exists():
            with open(self.schema_path, "r", encoding="utf-8") as f:
                conn.executescript(f.read())
            conn.commit()
        return conn

    def _execute_job(self, conn: sqlite3.Connection, job: _WriteJob) -> None:
        try:
            cur = conn.execute(job.sql, job.params)
            conn.commit()
            if job.result_event is not None:
                job.result_holder["rowid"] = cur.lastrowid
        except sqlite3.Error as exc:
            logger.error("DB write failed: %s | sql=%s", exc, job.sql)
            if job.result_event is not None:
                job.result_holder["error"] = str(exc)
        finally:
            if job.result_event is not None:
                job.result_event.set()


def open_read_connection(db_path: str) -> sqlite3.Connection:
    """Read-only-by-convention connection for readers (bridge_api,
    benchmarks). WAL mode means this can run concurrently with
    AsyncDBWriter's connection without lock errors."""
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=str)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())