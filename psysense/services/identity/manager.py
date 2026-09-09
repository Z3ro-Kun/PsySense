"""
services/identity/manager.py

IdentityManager is the single entry point for everything identity-related:
  - enroll_student(): register a new student from one or more images,
    maintaining embedding *history* (never overwriting).
  - resolve(): given a tracker_id + face crop, return the persistent
    student identity for this frame (cache-first, vector-search fallback).
  - handle_duplicate / quality gating / cache TTL are all applied here so
    every caller (Main pipeline, benchmarks, tests) gets the same
    guarantees without re-implementing them.

Depends only on the interfaces in core/interfaces.py (QualityAssessor,
EmbeddingExtractor, VectorIndex) — concrete backends are injected, which
is what makes this testable with mocks and swappable (e.g. FAISS <->
linear) without touching this file.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID, uuid4

import numpy as np

from core.exceptions import DuplicateEnrollmentError
from core.interfaces import EmbeddingExtractor, QualityAssessor, VectorIndex
from core.models import (
    FaceEmbedding,
    FaceQualityReport,
    IdentityMatchResult,
    MatchStatus,
    Student,
)
from services.identity.cache import ActiveIdentityCache
from utils import encode_image_bytes

logger = logging.getLogger("psysense.identity.manager")


class IdentityManager:
    def __init__(
        self,
        embedder: EmbeddingExtractor,
        quality_assessor: QualityAssessor,
        vector_index: VectorIndex,
        cache: Optional[ActiveIdentityCache] = None,
        match_distance_threshold: float = 0.42,
        duplicate_distance_threshold: float = 0.08,
        max_embeddings_per_student: int = 20,
        repository=None,  # db.repository.repository.Repository; optional (untyped to avoid a hard DB dependency)
        auto_update_enabled: bool = False,
        auto_update_distance_threshold: float = 0.20,
        auto_update_min_seconds_between_candidates: float = 3600.0,
    ):
        """
        match_distance_threshold: cosine-distance ceiling for accepting a
            vector-search hit as a positive identity match. Tune via the
            identity-accuracy benchmark (Phase 4) against real enrollment
            data — this default is a reasonable ArcFace starting point,
            not a guarantee.
        duplicate_distance_threshold: below this distance, a *new*
            enrollment image is considered a near-duplicate of an existing
            one for the same student and is skipped rather than bloating
            the embedding history.
        auto_update_*: human-in-the-loop profile enrichment (see
            _maybe_queue_auto_update_candidate) -- disabled by default
            since it requires a repository capable of storing raw images
            (bridge_api's throwaway enroll-only instance shouldn't queue
            these; only the pipeline's real IdentityManager should).
        """
        self._embedder = embedder
        self._quality = quality_assessor
        self._index = vector_index
        self._cache = cache or ActiveIdentityCache()
        self._match_threshold = match_distance_threshold
        self._dup_threshold = duplicate_distance_threshold
        self._max_embeddings_per_student = max_embeddings_per_student
        self._auto_update_enabled = auto_update_enabled
        self._auto_update_threshold = auto_update_distance_threshold
        self._auto_update_min_seconds = auto_update_min_seconds_between_candidates

        # In-memory embedding history; a persistence-backed repository
        # (Phase 3 DB layer) is injected/synced separately so this class
        # doesn't take a hard dependency on SQLite.
        self._students: dict[UUID, Student] = {}
        self._embeddings_by_student: dict[UUID, list[FaceEmbedding]] = {}
        self._repository = repository
        self._loaded_embedding_ids: set[UUID] = set()

    # ------------------------------------------------------------------ #
    # Persistence (Phase 3): rehydrate on startup, persist on enroll
    # ------------------------------------------------------------------ #

    def load_from_repository(self) -> None:
        """Rebuild in-memory state + vector index from the database. Call
        this once at startup, after construction, before serving traffic.
        Safe to call with an empty database (no-op)."""
        if self._repository is None:
            return
        for student in self._repository.get_all_students():
            self._students[student.student_id] = student
            self._embeddings_by_student.setdefault(student.student_id, [])

        for embedding in self._repository.get_all_embeddings():
            if embedding.student_id not in self._students:
                logger.warning("Embedding %s references unknown student %s — skipping",
                                embedding.embedding_id, embedding.student_id)
                continue
            self._embeddings_by_student[embedding.student_id].append(embedding)
            self._index.add(embedding.student_id, embedding.embedding_id, embedding.as_array())
            self._loaded_embedding_ids.add(embedding.embedding_id)

        logger.info(
            "IdentityManager rehydrated: %d students, %d embeddings",
            len(self._students), sum(len(v) for v in self._embeddings_by_student.values()),
        )

    def refresh_from_repository(self) -> int:
        """Incremental version of load_from_repository(): picks up students
        and embeddings written by *another process* since the last load (or
        the last refresh) -- e.g. a student enrolled through bridge_api
        while this pipeline process is already running -- without
        re-adding embeddings already in the vector index (which would
        create duplicate rows and skew search results) or reprocessing the
        whole table on every call. Also reconciles the other direction: a
        student deleted via bridge_api (Repository.delete_student()) must
        stop being recognized here too, not just disappear from listings.

        Call this periodically (main.py's periodic maintenance loop, not
        every frame) rather than only at startup. Returns the number of
        newly-added embeddings.
        """
        if self._repository is None:
            return 0

        current_students = {s.student_id: s for s in self._repository.get_all_students()}

        removed = 0
        for existing_id in list(self._students.keys()):
            if existing_id not in current_students:
                self._remove_student_local(existing_id)
                removed += 1

        for student_id, student in current_students.items():
            if student_id not in self._students:
                self._students[student_id] = student
                self._embeddings_by_student.setdefault(student_id, [])
            else:
                # Pick up display_name/active edits made via the API too.
                self._students[student_id] = student

        added = 0
        for embedding in self._repository.get_all_embeddings():
            if embedding.embedding_id in self._loaded_embedding_ids:
                continue
            if embedding.student_id not in self._students:
                logger.warning("Embedding %s references unknown student %s — skipping",
                                embedding.embedding_id, embedding.student_id)
                continue
            self._embeddings_by_student.setdefault(embedding.student_id, []).append(embedding)
            self._index.add(embedding.student_id, embedding.embedding_id, embedding.as_array())
            self._loaded_embedding_ids.add(embedding.embedding_id)
            added += 1

        if added or removed:
            logger.info("IdentityManager incremental refresh: +%d embeddings, -%d students, %d students total",
                        added, removed, len(self._students))
        return added

    def remove_student(self, student_id: UUID) -> None:
        """Public entry point for callers that hold a live IdentityManager
        and need a student removed immediately (rather than waiting for
        the next periodic refresh_from_repository() cycle) -- e.g. a
        single-process deployment where the same IdentityManager instance
        both resolves and handles admin actions."""
        self._remove_student_local(student_id)

    def _remove_student_local(self, student_id: UUID) -> None:
        self._students.pop(student_id, None)
        removed_embeddings = self._embeddings_by_student.pop(student_id, [])
        for emb in removed_embeddings:
            self._loaded_embedding_ids.discard(emb.embedding_id)
        self._index.remove_student(student_id)

    # ------------------------------------------------------------------ #
    # Enrollment
    # ------------------------------------------------------------------ #

    def enroll_student(
        self,
        display_name: str,
        images_bgr: list[np.ndarray],
        external_ref: Optional[str] = None,
        student_id: Optional[UUID] = None,
    ) -> tuple[Student, list[FaceEmbedding], list[str]]:
        """
        Enroll a student from one or more face images. Supports incremental
        re-enrollment: pass an existing student_id to add more images to an
        already-enrolled student rather than creating a duplicate record.

        Returns (student, accepted_embeddings, skipped_reasons).
        """
        student = self._students.get(student_id) if student_id else None
        is_new_student = student is None
        if student is None:
            student = Student(student_id=student_id or uuid4(), display_name=display_name, external_ref=external_ref)
            self._students[student.student_id] = student
            self._embeddings_by_student.setdefault(student.student_id, [])

        if is_new_student and self._repository is not None:
            self._repository.upsert_student(student)

        accepted: list[FaceEmbedding] = []
        skipped: list[str] = []

        for img in images_bgr:
            face = self._embedder.analyze(img) if hasattr(self._embedder, "analyze") else None
            landmarks = getattr(face, "kps", None) if face is not None else None
            box = self._box_from_face(face, img.shape) if face is not None else (0, 0, img.shape[1], img.shape[0])

            quality = self._quality.assess(img, box, landmarks_5pt=landmarks)
            if not quality.passed:
                skipped.append(f"quality_rejected:{[r.value for r in quality.reasons]}")
                continue

            vector = self._embedder.embed(img)
            if vector is None:
                skipped.append("embedding_extraction_failed")
                continue

            try:
                self._reject_if_duplicate(student.student_id, vector)
            except DuplicateEnrollmentError as exc:
                skipped.append(str(exc))
                continue

            embedding = FaceEmbedding(student_id=student.student_id, vector=vector.tolist(), quality=quality)
            self._index.add(student.student_id, embedding.embedding_id, vector)
            self._embeddings_by_student[student.student_id].append(embedding)
            self._loaded_embedding_ids.add(embedding.embedding_id)
            accepted.append(embedding)
            if self._repository is not None:
                self._repository.insert_embedding(embedding)

        self._enforce_history_cap(student.student_id)
        logger.info(
            "Enrolled student %s (%s): +%d embeddings, %d skipped",
            student.display_name, student.student_id, len(accepted), len(skipped),
        )
        return student, accepted, skipped

    def _reject_if_duplicate(self, student_id: UUID, vector: np.ndarray) -> None:
        existing = self._embeddings_by_student.get(student_id, [])
        for emb in existing:
            distance = 1.0 - float(np.dot(vector, emb.as_array()))
            if distance < self._dup_threshold:
                raise DuplicateEnrollmentError(
                    f"duplicate_of:{emb.embedding_id} distance={distance:.4f}"
                )

    def _enforce_history_cap(self, student_id: UUID) -> None:
        """Keep embedding history bounded per student — oldest low-quality
        embeddings are dropped first once the cap is exceeded, keeping the
        most recent + highest quality representatives."""
        embeddings = self._embeddings_by_student.get(student_id, [])
        if len(embeddings) <= self._max_embeddings_per_student:
            return
        embeddings.sort(key=lambda e: (e.quality.blur_score, e.created_at), reverse=True)
        keep = embeddings[: self._max_embeddings_per_student]
        dropped_count = len(embeddings) - len(keep)
        if dropped_count > 0:
            self._index.remove_student(student_id)  # flat index has no per-vector delete; rebuild with kept set
            for emb in keep:
                self._index.add(student_id, emb.embedding_id, emb.as_array())
            self._embeddings_by_student[student_id] = keep
            if self._repository is not None:
                # Keep the durable copy in sync: drop all rows for this
                # student, re-insert only the kept ones. Infrequent (only
                # triggered when the cap is exceeded), so the extra writes
                # are cheap relative to how rarely this runs.
                self._repository.delete_embeddings_for_student(student_id)
                for emb in keep:
                    self._repository.insert_embedding(emb)
            logger.info("History cap enforced for student %s: dropped %d oldest/lowest-quality embeddings",
                        student_id, dropped_count)

    # ------------------------------------------------------------------ #
    # Resolution (per-frame identity lookup)
    # ------------------------------------------------------------------ #

    def resolve(
        self, tracker_id: int, face_bgr: np.ndarray, face_box: tuple[int, int, int, int],
        session_id: Optional[str] = None,
    ) -> IdentityMatchResult:
        """
        Resolve the persistent identity for a tracked face this frame.
        Cache-first; falls back to embedding + vector search on cache miss,
        cache expiry, or the periodic forced re-verification the cache
        triggers to guard against silent tracker-ID identity swaps.

        session_id is used only for a very-high-confidence match: see
        _maybe_queue_auto_update_candidate.
        """
        import time

        start = time.perf_counter()

        cached = self._cache.get(tracker_id)
        if cached is not None:
            return IdentityMatchResult(
                status=MatchStatus.CACHE_HIT,
                student_id=cached.student_id,
                confidence=cached.confidence,
                tracker_id=tracker_id,
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )

        face = self._embedder.analyze(face_bgr) if hasattr(self._embedder, "analyze") else None
        landmarks = getattr(face, "kps", None) if face is not None else None
        # Use the actually-detected face box for quality assessment, not
        # whatever region the caller passed in as face_box -- main.py's
        # caller passes the full YOLO person-box, which would make
        # blur/brightness measure the wrong thing (see quality.py). This
        # mirrors enroll_student()'s identical derivation below, so the
        # same person is assessed consistently at enroll time and at
        # resolve time.
        detected_face_box = self._box_from_face(face, face_bgr.shape) if face is not None else face_box
        quality = self._quality.assess(face_bgr, detected_face_box, landmarks_5pt=landmarks)
        if not quality.passed:
            logger.debug(
                "Identity resolve: quality rejected for tracker %s (reasons=%s, blur=%.1f, brightness=%.1f, yaw=%s, occlusion=%.2f)",
                tracker_id, [r.value for r in quality.reasons], quality.blur_score, quality.brightness,
                quality.yaw_deg, quality.occlusion_score,
            )
            return IdentityMatchResult(
                status=MatchStatus.REJECTED_QUALITY,
                quality=quality,
                tracker_id=tracker_id,
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )

        vector = self._embedder.embed(face_bgr)
        if vector is None:
            logger.debug("Identity resolve: embedding extraction failed for tracker %s", tracker_id)
            return IdentityMatchResult(
                status=MatchStatus.UNKNOWN,
                quality=quality,
                tracker_id=tracker_id,
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )

        hits = self._index.search(vector, top_k=1)
        if not hits or hits[0][2] > self._match_threshold:
            logger.debug(
                "Identity resolve: no match for tracker %s (best_distance=%s, threshold=%.3f, index_size=%d)",
                tracker_id, hits[0][2] if hits else None, self._match_threshold, self._index.size(),
            )
            return IdentityMatchResult(
                status=MatchStatus.UNKNOWN,
                distance=hits[0][2] if hits else None,
                quality=quality,
                tracker_id=tracker_id,
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )

        student_id, _embedding_id, distance = hits[0]
        confidence = max(0.0, 1.0 - distance)
        self._cache.put(tracker_id, student_id, confidence)

        if self._auto_update_enabled and distance <= self._auto_update_threshold:
            self._maybe_queue_auto_update_candidate(
                student_id=student_id, face_bgr=face_bgr, vector=vector, quality=quality,
                distance=distance, tracker_id=tracker_id, session_id=session_id,
            )

        return IdentityMatchResult(
            status=MatchStatus.MATCHED,
            student_id=student_id,
            distance=distance,
            confidence=confidence,
            quality=quality,
            tracker_id=tracker_id,
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )

    def _maybe_queue_auto_update_candidate(
        self, student_id: UUID, face_bgr: np.ndarray, vector: np.ndarray, quality: FaceQualityReport,
        distance: float, tracker_id: int, session_id: Optional[str],
    ) -> None:
        """Human-in-the-loop profile enrichment: NEVER adds an embedding
        to a profile by itself. Only queues a candidate (raw image +
        embedding) for an admin to explicitly approve/reject in the
        dashboard's Review page -- see api/routers/identity_candidates.py.
        Only fires on matches already far more confident than the normal
        matching threshold (caller checks this), specifically so a human
        reviewer is confirming a near-certain case, not adjudicating a
        borderline one."""
        if self._repository is None:
            return
        try:
            since = datetime.now(timezone.utc) - timedelta(seconds=self._auto_update_min_seconds)
            recent = self._repository.count_recent_pending_candidates_for_student(student_id, since=since)
            if recent > 0:
                return  # rate-limited -- already queued one for this student recently

            image_bytes = encode_image_bytes(face_bgr)
            if image_bytes is None:
                return

            self._repository.insert_pending_candidate(
                student_id=student_id, vector=vector, image_jpeg=image_bytes, quality=quality,
                distance=distance, tracker_id=tracker_id, session_id=session_id,
            )
            logger.info(
                "Queued auto-update candidate for review",
                extra={"context": {"student_id": str(student_id), "distance": distance, "tracker_id": tracker_id}},
            )
        except Exception as exc:  # noqa: BLE001 -- never let review-queue bookkeeping break resolve()
            logger.error("Failed to queue auto-update candidate: %s", exc)

    def on_tracker_dropped(self, tracker_id: int) -> None:
        """Call when the tracker retires an ID (person left frame / lost
        for too long) so the cache doesn't hold a stale mapping ready to be
        silently reused for a different physical person who gets the same
        tracker ID later."""
        self._cache.drop(tracker_id)

    def sweep_cache(self) -> int:
        return self._cache.sweep()

    def get_student(self, student_id: UUID) -> Optional[Student]:
        """Look up a student's record (e.g. display_name) for UI/debug
        overlay purposes. Returns None if unknown."""
        return self._students.get(student_id)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _box_from_face(face, img_shape) -> tuple[int, int, int, int]:
        if face is None or getattr(face, "bbox", None) is None:
            h, w = img_shape[:2]
            return (0, 0, w, h)
        x1, y1, x2, y2 = face.bbox
        return int(x1), int(y1), int(x2), int(y2)