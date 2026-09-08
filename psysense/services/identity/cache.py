"""
services/identity/cache.py

Active Identity Cache: once a tracker ID has been resolved to a student
identity via vector search, subsequent frames for that same tracker ID
skip the vector search entirely and reuse the cached result until it
expires or the tracker ID is dropped.

This is explicitly NOT the source of truth for identity — it's a
performance shortcut keyed on a short-lived tracker ID. It must never be
consulted across a tracker-ID switch (occlusion, re-entry, ID reuse by
the tracker) without re-verification; see IdentityManager.resolve() for
how re-verification is triggered.

Fixes the unbounded-growth bug present in the original pose_analyzer.py
history dict: every entry carries a last_seen timestamp and is evicted
by TTL, plus there's an explicit sweep() the pipeline can call
periodically instead of relying only on lazy eviction.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Optional
from uuid import UUID


@dataclass
class CacheEntry:
    student_id: UUID
    confidence: float
    last_seen: float
    hits: int = 0
    frames_since_reverify: int = 0


class ActiveIdentityCache:
    def __init__(self, ttl_seconds: float = 30.0, reverify_every_n_frames: int = 150):
        """
        ttl_seconds: entries not touched within this window are evicted.
        reverify_every_n_frames: even on a cache hit, force a fresh vector
            search every N lookups for a given tracker ID. This bounds the
            damage from a silent tracker-ID identity swap (two people
            crossing paths, tracker mis-association) instead of trusting a
            single early match forever.
        """
        self._ttl = ttl_seconds
        self._reverify_every = reverify_every_n_frames
        self._entries: dict[int, CacheEntry] = {}
        self._lock = Lock()

    def get(self, tracker_id: int) -> Optional[CacheEntry]:
        with self._lock:
            entry = self._entries.get(tracker_id)
            if entry is None:
                return None
            now = time.time()
            if now - entry.last_seen > self._ttl:
                del self._entries[tracker_id]
                return None
            entry.last_seen = now
            entry.hits += 1
            entry.frames_since_reverify += 1
            if entry.frames_since_reverify >= self._reverify_every:
                # Signal caller to re-verify by not returning a stale hit;
                # caller (IdentityManager) treats this as a cache miss.
                entry.frames_since_reverify = 0
                return None
            return entry

    def put(self, tracker_id: int, student_id: UUID, confidence: float) -> None:
        with self._lock:
            self._entries[tracker_id] = CacheEntry(
                student_id=student_id, confidence=confidence, last_seen=time.time()
            )

    def drop(self, tracker_id: int) -> None:
        with self._lock:
            self._entries.pop(tracker_id, None)

    def sweep(self) -> int:
        """Evict all expired entries. Returns number removed. Call this
        periodically (e.g. once per second) from the main loop rather than
        relying solely on lazy per-lookup eviction."""
        now = time.time()
        with self._lock:
            expired = [tid for tid, e in self._entries.items() if now - e.last_seen > self._ttl]
            for tid in expired:
                del self._entries[tid]
            return len(expired)

    def size(self) -> int:
        with self._lock:
            return len(self._entries)