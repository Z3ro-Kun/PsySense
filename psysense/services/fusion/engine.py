"""
services/fusion/engine.py

Fusion Engine: the single place raw per-frame emotion and pose samples
are combined with the current resolved identity into a FusionEvent
before anything touches the database. Nothing else should call
Repository.insert_emotion_event / insert_pose_event / insert_behavior_event
directly -- they all go through here, so windowing/averaging and
behavior-flag logic exist in exactly one place.

Save as: services/fusion/engine.py

Why a windowed engine rather than writing every raw sample:
  - a single frame's emotion/pose reading is noisy; a short rolling
    window averaged before writing is both a data-quality improvement
    and a huge reduction in write volume compared to the old
    write-every-10-seconds-per-track approach in the original Main.py.
  - behavior flags (e.g. "sustained slump") are inherently about
    *duration*, not a single sample, so they can only be computed here,
    after windowing.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from core.models import BehaviorFlag, BehaviorSeverity, FusionEvent, PoseReading
from logging_.logger import get_logger

logger = get_logger("fusion.engine")


@dataclass
class _Sample:
    ts: float
    emotions: dict[str, float]
    pose: PoseReading


@dataclass
class _TrackBuffer:
    student_id: Optional[UUID]
    tracker_id: int
    identity_confidence: Optional[float]
    session_id: Optional[str]
    samples: deque = field(default_factory=lambda: deque(maxlen=600))  # ~ bounded, not unbounded like the old pose_analyzer history
    slump_streak_start: Optional[float] = None


class FusionEngine:
    def __init__(
        self,
        window_seconds: float = 10.0,
        slump_threshold: float = 0.6,
        slump_sustained_seconds: float = 30.0,
        negative_emotion_threshold: float = 0.55,
        negative_emotions: tuple[str, ...] = ("sad", "angry", "fear", "disgust"),
    ):
        self._window_seconds = window_seconds
        self._slump_threshold = slump_threshold
        self._slump_sustained_seconds = slump_sustained_seconds
        self._negative_threshold = negative_emotion_threshold
        self._negative_emotions = negative_emotions
        self._buffers: dict[int, _TrackBuffer] = {}

    # ------------------------------------------------------------------ #
    def observe(
        self,
        tracker_id: int,
        student_id: Optional[UUID],
        identity_confidence: Optional[float],
        session_id: Optional[str],
        emotions: Optional[dict[str, float]] = None,
        pose: Optional[PoseReading] = None,
    ) -> None:
        """Feed one raw sample (emotion and/or pose -- either may be None if
        that modality wasn't sampled this cycle, since detection/tracking,
        identity, emotion, and pose all run at independent frequencies)."""
        buf = self._buffers.get(tracker_id)
        if buf is None or buf.student_id != student_id:
            # New track, or identity was just re-resolved to someone
            # different (tracker-ID reuse) -- start a fresh buffer rather
            # than silently blending two different people's samples.
            buf = _TrackBuffer(
                student_id=student_id, tracker_id=tracker_id,
                identity_confidence=identity_confidence, session_id=session_id,
            )
            self._buffers[tracker_id] = buf
        buf.identity_confidence = identity_confidence
        buf.session_id = session_id
        buf.samples.append(_Sample(ts=time.time(), emotions=emotions or {}, pose=pose or PoseReading()))

    def on_tracker_dropped(self, tracker_id: int) -> Optional[FusionEvent]:
        """Flush and remove the buffer for a tracker that's gone (person
        left frame / tracking lost). Returns a final FusionEvent if there
        was unflushed data, else None."""
        buf = self._buffers.pop(tracker_id, None)
        if buf is None or not buf.samples:
            return None
        return self._build_event(buf, list(buf.samples))

    def flush_due(self) -> list[FusionEvent]:
        """Call periodically (e.g. once per second) from the main loop.
        Emits one FusionEvent per track whose oldest unflushed sample has
        aged past window_seconds, then clears that track's buffer."""
        events: list[FusionEvent] = []
        now = time.time()
        for tracker_id, buf in list(self._buffers.items()):
            if not buf.samples:
                continue
            oldest = buf.samples[0].ts
            if now - oldest >= self._window_seconds:
                events.append(self._build_event(buf, list(buf.samples)))
                buf.samples.clear()
        return events

    # ------------------------------------------------------------------ #
    def _build_event(self, buf: _TrackBuffer, samples: list[_Sample]) -> FusionEvent:
        avg_emotions = self._average_emotions(samples)
        avg_pose = self._average_pose(samples)
        flags = self._detect_behavior_flags(buf, avg_emotions, avg_pose, samples)

        event = FusionEvent(
            student_id=buf.student_id,
            tracker_id=buf.tracker_id,
            session_id=buf.session_id,
            emotions=avg_emotions,
            pose=avg_pose,
            identity_confidence=buf.identity_confidence,
            behavior_flags=flags,
            sample_count=len(samples),
            window_start=datetime.fromtimestamp(samples[0].ts, tz=timezone.utc),
            window_end=datetime.fromtimestamp(samples[-1].ts, tz=timezone.utc),
        )
        if flags:
            logger.info(
                "Behavior flags raised",
                extra={"context": {"tracker_id": buf.tracker_id, "student_id": str(buf.student_id),
                                    "flags": [f.event_type for f in flags]}},
            )
        return event

    @staticmethod
    def _average_emotions(samples: list[_Sample]) -> dict[str, float]:
        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for s in samples:
            for k, v in s.emotions.items():
                totals[k] += v
                counts[k] += 1
        if not totals:
            return {}
        avg = {k: totals[k] / counts[k] for k in totals}
        total = sum(avg.values()) or 1.0
        return {k: v / total for k, v in avg.items()}

    @staticmethod
    def _average_pose(samples: list[_Sample]) -> PoseReading:
        poses = [s.pose for s in samples]
        n = len(poses) or 1
        return PoseReading(
            slumped_score=sum(p.slumped_score for p in poses) / n,
            rigidity_score=sum(p.rigidity_score for p in poses) / n,
            fidgeting_score=sum(p.fidgeting_score for p in poses) / n,
        )

    def _detect_behavior_flags(
        self, buf: _TrackBuffer, avg_emotions: dict[str, float], avg_pose: PoseReading, samples: list[_Sample],
    ) -> list[BehaviorFlag]:
        flags: list[BehaviorFlag] = []

        # Sustained slump: track how long slumped_score has stayed above
        # threshold across consecutive windows, not just this one window.
        if avg_pose.slumped_score >= self._slump_threshold:
            if buf.slump_streak_start is None:
                buf.slump_streak_start = samples[0].ts
            streak = samples[-1].ts - buf.slump_streak_start
            if streak >= self._slump_sustained_seconds:
                flags.append(BehaviorFlag(
                    event_type="prolonged_slump",
                    severity=BehaviorSeverity.WARNING,
                    detail=f"Slumped posture sustained for {streak:.0f}s (score={avg_pose.slumped_score:.2f})",
                ))
        else:
            buf.slump_streak_start = None

        # Elevated negative affect within this window.
        negative_total = sum(avg_emotions.get(e, 0.0) for e in self._negative_emotions)
        if negative_total >= self._negative_threshold:
            flags.append(BehaviorFlag(
                event_type="elevated_negative_affect",
                severity=BehaviorSeverity.INFO,
                detail=f"Combined negative-emotion score {negative_total:.2f} in this window",
            ))

        # Combined signal: negative affect + slumped posture together is a
        # stronger signal than either alone.
        if negative_total >= self._negative_threshold and avg_pose.slumped_score >= self._slump_threshold:
            flags.append(BehaviorFlag(
                event_type="disengagement_signal",
                severity=BehaviorSeverity.WARNING,
                detail="Elevated negative affect co-occurring with slumped posture",
            ))

        return flags