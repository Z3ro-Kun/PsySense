"""
main.py

Save as: main.py (project root — replaces the old Main.py)

Orchestrates the full pipeline: detection + tracking run every frame;
identity resolution, emotion analysis, and pose analysis each run at
their own configured interval per tracked person (config.yaml ->
pipeline.frequencies), not every frame. Everything is fed into the
Fusion Engine, which is the only thing that writes to the database
(through the single async writer, never directly).

This intentionally does NOT open a display window by default
(pipeline.display.headless: true) -- the original Main.py's tight
coupling to cv2.imshow made it unusable as a background/server process.
Set headless: false in config.yaml for a debug window during development.

Run:
    python main.py
"""
from __future__ import annotations

import base64
import signal
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests

from core.config import AppConfig, load_config
from core.models import BehaviorSeverity, PoseReading
from db.repository.repository import Repository
from db.writer import AsyncDBWriter
from logging_.logger import get_logger
from services.fusion.engine import FusionEngine
from services.identity.cache import ActiveIdentityCache
from services.identity.embedder import get_embedder
from services.identity.manager import IdentityManager
from services.identity.quality import HeuristicQualityAssessor
from services.identity.vector_index import build_vector_index
from services.tracking.tracker import build_tracker
from utils import crop_with_padding, encode_image_b64

logger = get_logger("main")


@dataclass
class TrackState:
    """Per-tracker-id bookkeeping for independent-frequency scheduling.
    Deliberately NOT where identity lives -- student_id here is a cached
    convenience updated by IdentityManager.resolve(), never authoritative
    on its own (see services/identity/manager.py)."""

    tracker_id: int
    session_id: str
    student_id: Optional[uuid.UUID] = None
    identity_confidence: Optional[float] = None
    last_identity_resolve: float = 0.0
    last_emotion: float = 0.0
    last_pose: float = 0.0
    last_seen: float = field(default_factory=time.time)
    # Latest known values purely for the debug overlay -- NOT the source
    # of truth (that's the Fusion Engine's windowed buffers). Updated
    # from the executor threads in _fetch_emotion/_fetch_pose; simple
    # attribute replacement is safe under the GIL for this display-only
    # purpose without needing a lock.
    last_box: Optional[tuple[int, int, int, int]] = None
    display_emotions: dict = field(default_factory=dict)
    display_pose: PoseReading = field(default_factory=PoseReading)


class PsySensePipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="psysense-io")
        self._track_states: dict[int, TrackState] = {}
        self._running = False

        self._writer = AsyncDBWriter(
            db_path=config.pipeline.database.path,
            schema_path=str(Path(__file__).parent / "db" / "schema.sql"),
        )
        self._writer.start()
        self._repository = Repository(self._writer, config.pipeline.database.path)

        logger.info("Loading models (YOLO, ArcFace) -- this can take a while on first run...")
        from ultralytics import YOLO
        self._yolo = YOLO(config.pipeline.detection.yolo_weights)

        embedder = get_embedder(
            ctx_id=config.identity.embedder.ctx_id,
            det_size=tuple(config.identity.embedder.det_size),
        )
        quality = HeuristicQualityAssessor(
            blur_threshold=config.identity.quality.blur_threshold,
            brightness_range=(config.identity.quality.brightness_min, config.identity.quality.brightness_max),
            max_yaw_deg=config.identity.quality.max_yaw_deg,
            max_occlusion=config.identity.quality.max_occlusion,
            min_face_size_px=config.identity.quality.min_face_size_px,
        )
        vector_index = build_vector_index(
            dim=config.identity.vector_index.dim,
            prefer_faiss=(config.identity.vector_index.backend != "linear"),
        )
        self._identity = IdentityManager(
            embedder=embedder,
            quality_assessor=quality,
            vector_index=vector_index,
            cache=ActiveIdentityCache(
                ttl_seconds=config.identity.cache.ttl_seconds,
                reverify_every_n_frames=config.identity.cache.reverify_every_n_frames,
            ),
            match_distance_threshold=config.identity.matching.match_distance_threshold,
            duplicate_distance_threshold=config.identity.matching.duplicate_distance_threshold,
            max_embeddings_per_student=config.identity.matching.max_embeddings_per_student,
            repository=self._repository,
        )
        self._identity.load_from_repository()

        tracker_cfg = config.pipeline.tracker
        if tracker_cfg.kind == "bytetrack":
            self._tracker = build_tracker("bytetrack", **tracker_cfg.bytetrack.model_dump())
        elif tracker_cfg.kind == "strongsort":
            self._tracker = build_tracker("strongsort", **tracker_cfg.strongsort.model_dump())
        else:
            raise ValueError(f"Unknown tracker.kind in config.yaml: {tracker_cfg.kind}")
        logger.info(f"Tracker initialized: {self._tracker.name}")

        self._fusion = FusionEngine(window_seconds=config.pipeline.frequencies.fusion_flush_interval_sec)

        self._last_cache_sweep = time.time()
        self._cap: Optional[cv2.VideoCapture] = None

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        self._running = True
        signal.signal(signal.SIGINT, lambda *_: self._request_stop())
        signal.signal(signal.SIGTERM, lambda *_: self._request_stop())

        cam_cfg = self.config.pipeline.camera
        self._cap = self._open_camera(cam_cfg.source)
        consecutive_failures = 0

        logger.info("Pipeline started", extra={"context": {"camera": cam_cfg.source, "tracker": self._tracker.name}})

        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                consecutive_failures += 1
                logger.warning(f"Camera read failed ({consecutive_failures}/{cam_cfg.max_consecutive_read_failures})")
                if consecutive_failures >= cam_cfg.max_consecutive_read_failures:
                    logger.error("Too many consecutive camera failures -- attempting reconnect")
                    self._cap.release()
                    time.sleep(cam_cfg.reconnect_delay_seconds)
                    try:
                        self._cap = self._open_camera(cam_cfg.source)
                        consecutive_failures = 0
                    except Exception as exc:  # noqa: BLE001
                        logger.error(f"Camera reconnect failed: {exc}")
                        time.sleep(cam_cfg.reconnect_delay_seconds)
                continue
            consecutive_failures = 0

            try:
                self._process_frame(frame)
            except Exception as exc:  # noqa: BLE001
                # A single bad frame must never kill the whole pipeline --
                # this is the top-level safety net the original Main.py
                # didn't have.
                logger.error(f"Unhandled error processing frame: {exc}", exc_info=True)

            self._run_periodic_maintenance()

            if not self.config.pipeline.display.headless:
                display_frame = self._draw_debug_overlay(frame)
                cv2.imshow("PsySense (debug)", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._request_stop()

        self._shutdown()

    def _request_stop(self) -> None:
        logger.info("Shutdown requested")
        self._running = False

    # ------------------------------------------------------------------ #
    def _open_camera(self, source) -> cv2.VideoCapture:
        cap_source = int(source) if isinstance(source, str) and source.isdigit() else source
        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera/video source: {source}")
        return cap

    # ------------------------------------------------------------------ #
    def _process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        dets = self._detect_persons(frame)

        try:
            tracks = self._tracker.update(dets, frame)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Tracker update failed: {exc}")
            return None

        seen_ids = set()
        now = time.time()

        for row in tracks:
            x1, y1, x2, y2 = row[:4].astype(int)
            tracker_id = int(row[4])
            seen_ids.add(tracker_id)

            state = self._track_states.get(tracker_id)
            if state is None:
                session_id = str(uuid.uuid4())
                state = TrackState(tracker_id=tracker_id, session_id=session_id)
                self._track_states[tracker_id] = state
                self._repository.open_session(
                    session_id=session_id, tracker_id=tracker_id, student_id=None,
                    camera_id=self.config.pipeline.camera.camera_id,
                )
            state.last_seen = now
            state.last_box = (int(x1), int(y1), int(x2), int(y2))

            person_crop = crop_with_padding(frame, x1, y1, x2, y2)
            if person_crop is None or person_crop.size == 0:
                continue

            freqs = self.config.pipeline.frequencies

            if now - state.last_identity_resolve >= freqs.identity_resolve_interval_sec:
                self._resolve_identity(state, person_crop, (x1, y1, x2, y2))
                state.last_identity_resolve = now

            if now - state.last_emotion >= freqs.emotion_interval_sec:
                self._executor.submit(self._fetch_emotion, state, person_crop)
                state.last_emotion = now

            if now - state.last_pose >= freqs.pose_interval_sec:
                self._executor.submit(self._fetch_pose, state, person_crop)
                state.last_pose = now

        # Any tracker ID we previously knew about but didn't see this
        # frame's tracks: not yet dropped (tracker keeps IDs alive for a
        # buffer period internally), so we only clean up in
        # _run_periodic_maintenance based on last_seen staleness.
        _ = seen_ids  # currently informational; staleness handled below
        return tracks

    def _detect_persons(self, frame: np.ndarray) -> np.ndarray:
        det_cfg = self.config.pipeline.detection
        try:
            results = self._yolo(frame, verbose=False)[0]
        except Exception as exc:  # noqa: BLE001
            logger.error(f"YOLO detection failed: {exc}")
            return np.empty((0, 6), dtype=np.float32)

        dets = []
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls != det_cfg.person_class_id or conf < det_cfg.conf_threshold:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dets.append([x1, y1, x2, y2, conf, cls])
        return np.array(dets, dtype=np.float32) if dets else np.empty((0, 6), dtype=np.float32)

    # ------------------------------------------------------------------ #
    def _resolve_identity(self, state: TrackState, person_crop: np.ndarray, box: tuple[int, int, int, int]) -> None:
        try:
            result = self._identity.resolve(state.tracker_id, person_crop, box)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Identity resolution failed for tracker {state.tracker_id}: {exc}")
            return

        if result.student_id is not None and result.student_id != state.student_id:
            logger.info(
                "Identity resolved",
                extra={"context": {"tracker_id": state.tracker_id, "student_id": str(result.student_id),
                                    "status": result.status.value, "confidence": result.confidence}},
            )
        state.student_id = result.student_id
        state.identity_confidence = result.confidence

    def _fetch_emotion(self, state: TrackState, person_crop: np.ndarray) -> None:
        """Runs in the thread pool -- network I/O must never block the
        capture/detection/tracking loop, which is the bottleneck the
        original Main.py had (sequential per-track blocking HTTP calls)."""
        try:
            b64 = encode_image_b64(person_crop)
            resp = requests.post(
                self.config.pipeline.services.emotion_url,
                json={"image_b64": b64},
                timeout=self.config.pipeline.services.request_timeout_sec,
            )
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("status") == "success" and payload.get("data"):
                state.display_emotions = payload["data"]
                self._fusion.observe(
                    tracker_id=state.tracker_id, student_id=state.student_id,
                    identity_confidence=state.identity_confidence, session_id=state.session_id,
                    emotions=payload["data"],
                )
        except requests.RequestException as exc:
            logger.warning(f"Emotion service call failed for tracker {state.tracker_id}: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Unexpected error fetching emotion for tracker {state.tracker_id}: {exc}")

    def _fetch_pose(self, state: TrackState, person_crop: np.ndarray) -> None:
        try:
            b64 = encode_image_b64(person_crop)
            resp = requests.post(
                self.config.pipeline.services.pose_url,
                json={
                    "image_b64": b64,
                    "track_id": state.tracker_id,
                    "student_id": str(state.student_id) if state.student_id else None,
                },
                timeout=self.config.pipeline.services.request_timeout_sec,
            )
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("status") == "success" and payload.get("data"):
                d = payload["data"]
                pose_reading = PoseReading(
                    slumped_score=d.get("slumped_score", 0.0),
                    rigidity_score=d.get("rigidity_score", 0.0),
                    fidgeting_score=d.get("fidgeting_score", 0.0),
                )
                state.display_pose = pose_reading
                self._fusion.observe(
                    tracker_id=state.tracker_id, student_id=state.student_id,
                    identity_confidence=state.identity_confidence, session_id=state.session_id,
                    pose=pose_reading,
                )
        except requests.RequestException as exc:
            logger.warning(f"Pose service call failed for tracker {state.tracker_id}: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Unexpected error fetching pose for tracker {state.tracker_id}: {exc}")

    # ------------------------------------------------------------------ #
    def _run_periodic_maintenance(self) -> None:
        now = time.time()
        freqs = self.config.pipeline.frequencies

        for event in self._fusion.flush_due():
            self._persist_fusion_event(event)

        stale_track_ttl = 5.0  # seconds of no detections before considering a tracker id gone
        stale_ids = [tid for tid, s in self._track_states.items() if now - s.last_seen > stale_track_ttl]
        for tid in stale_ids:
            state = self._track_states.pop(tid)
            self._identity.on_tracker_dropped(tid)
            final_event = self._fusion.on_tracker_dropped(tid)
            if final_event is not None:
                self._persist_fusion_event(final_event)
            self._repository.close_session(state.session_id, status="closed")

        if now - self._last_cache_sweep >= freqs.cache_sweep_interval_sec:
            evicted = self._identity.sweep_cache()
            if evicted:
                logger.info(f"Active identity cache swept {evicted} expired entries")
            self._last_cache_sweep = now

    def _persist_fusion_event(self, event) -> None:
        if event.emotions:
            self._repository.insert_emotion_event(event.session_id, event.student_id, event.emotions)
        if event.pose.slumped_score or event.pose.rigidity_score or event.pose.fidgeting_score:
            self._repository.insert_pose_event(
                event.session_id, event.student_id,
                event.pose.slumped_score, event.pose.rigidity_score, event.pose.fidgeting_score,
            )
        for flag in event.behavior_flags:
            self._repository.insert_behavior_event(
                event.session_id, event.student_id, flag.event_type,
                {"detail": flag.detail}, severity=flag.severity.value,
            )

    # ------------------------------------------------------------------ #
    # Debug overlay (only used when pipeline.display.headless is false)
    # ------------------------------------------------------------------ #
    def _draw_debug_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draws, per tracked person: bounding box, tracker id, resolved
        identity (or 'unknown'), and the latest known emotion/pose
        readings. Reads only from TrackState's cached display_* fields --
        never blocks on or triggers new inference calls."""
        overlay = frame.copy()
        now = time.time()

        for state in self._track_states.values():
            if state.last_box is None or now - state.last_seen > 1.0:
                continue  # stale this frame, don't draw a ghost box
            x1, y1, x2, y2 = state.last_box

            if state.student_id is not None:
                student = self._identity.get_student(state.student_id)
                name = student.display_name if student else str(state.student_id)[:8]
                label = f"#{state.tracker_id} {name}"
                if state.identity_confidence is not None:
                    label += f" ({state.identity_confidence:.2f})"
                color = (0, 200, 0)  # green: matched
            else:
                label = f"#{state.tracker_id} unknown"
                color = (0, 165, 255)  # orange: unresolved/unmatched

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            lines = []
            if state.display_emotions:
                top_emotion = max(state.display_emotions, key=state.display_emotions.get)
                lines.append(f"{top_emotion} {state.display_emotions[top_emotion]:.2f}")
            p = state.display_pose
            if p.slumped_score or p.rigidity_score or p.fidgeting_score:
                lines.append(f"slump {p.slumped_score:.2f}  rigid {p.rigidity_score:.2f}  fidget {p.fidgeting_score:.2f}")

            for i, line in enumerate(lines):
                y = y2 + 18 + i * 18
                cv2.putText(overlay, line, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.putText(
            overlay, f"tracker={self._tracker.name}  people={len(self._track_states)}",
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
        return overlay

    # ------------------------------------------------------------------ #
    def _shutdown(self) -> None:
        logger.info("Shutting down pipeline...")
        for tid, state in list(self._track_states.items()):
            final_event = self._fusion.on_tracker_dropped(tid)
            if final_event is not None:
                self._persist_fusion_event(final_event)
            self._repository.close_session(state.session_id, status="closed")

        self._executor.shutdown(wait=True, cancel_futures=False)
        if self._cap is not None:
            self._cap.release()
        cv2.destroyAllWindows()
        self._writer.stop()
        self._repository.close()
        logger.info("Pipeline stopped cleanly")


def main() -> None:
    config = load_config()
    pipeline = PsySensePipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()