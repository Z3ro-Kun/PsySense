"""
tests/test_fusion_engine.py

Save as: tests/test_fusion_engine.py

Validates windowed averaging, tracker-ID buffer isolation on identity
change, and behavior-flag detection (sustained slump, negative affect,
combined disengagement signal) without needing any DB, camera, or model.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models import PoseReading  # noqa: E402
from services.fusion.engine import FusionEngine  # noqa: E402


def test_flush_due_emits_after_window_elapses():
    engine = FusionEngine(window_seconds=0.2)
    sid = uuid4()
    engine.observe(tracker_id=1, student_id=sid, identity_confidence=0.9, session_id="s1",
                    emotions={"happy": 1.0}, pose=PoseReading(slumped_score=0.1))

    assert engine.flush_due() == []  # window not elapsed yet
    time.sleep(0.25)
    events = engine.flush_due()
    assert len(events) == 1
    assert events[0].student_id == sid
    assert events[0].emotions["happy"] == 1.0


def test_averaging_across_multiple_samples():
    engine = FusionEngine(window_seconds=0.1)
    sid = uuid4()
    engine.observe(1, sid, 0.9, "s1", emotions={"happy": 1.0}, pose=PoseReading(slumped_score=0.2))
    engine.observe(1, sid, 0.9, "s1", emotions={"happy": 0.0, "sad": 1.0}, pose=PoseReading(slumped_score=0.6))
    time.sleep(0.15)
    events = engine.flush_due()
    assert len(events) == 1
    ev = events[0]
    assert ev.sample_count == 2
    assert abs(ev.pose.slumped_score - 0.4) < 1e-6
    # emotions renormalized to sum to 1
    assert abs(sum(ev.emotions.values()) - 1.0) < 1e-6


def test_identity_change_on_same_tracker_starts_fresh_buffer():
    """If a tracker ID gets re-resolved to a different student mid-stream
    (tracker mis-association corrected), old samples must not blend into
    the new person's averages."""
    engine = FusionEngine(window_seconds=5.0)
    sid_a, sid_b = uuid4(), uuid4()
    engine.observe(1, sid_a, 0.9, "s1", emotions={"happy": 1.0})
    engine.observe(1, sid_b, 0.9, "s2", emotions={"sad": 1.0})  # identity switched

    buf = engine._buffers[1]
    assert buf.student_id == sid_b
    assert len(buf.samples) == 1  # old sample discarded, not blended


def test_sustained_slump_flag_requires_duration_not_single_sample():
    engine = FusionEngine(window_seconds=0.05, slump_threshold=0.6, slump_sustained_seconds=0.15)
    sid = uuid4()

    # First window: slumped, but not long enough yet -> no flag
    engine.observe(1, sid, 0.9, "s1", pose=PoseReading(slumped_score=0.8))
    time.sleep(0.06)
    events = engine.flush_due()
    assert events
    assert not any(f.event_type == "prolonged_slump" for f in events[0].behavior_flags)

    # Keep observing slumped posture across more windows until the streak
    # exceeds slump_sustained_seconds
    flagged = False
    for _ in range(6):
        engine.observe(1, sid, 0.9, "s1", pose=PoseReading(slumped_score=0.8))
        time.sleep(0.06)
        for ev in engine.flush_due():
            if any(f.event_type == "prolonged_slump" for f in ev.behavior_flags):
                flagged = True
    assert flagged


def test_negative_affect_and_disengagement_flags():
    engine = FusionEngine(window_seconds=0.05, negative_emotion_threshold=0.5)
    sid = uuid4()
    engine.observe(1, sid, 0.9, "s1", emotions={"sad": 0.7, "happy": 0.3},
                    pose=PoseReading(slumped_score=0.9))
    time.sleep(0.06)
    events = engine.flush_due()
    assert events
    flag_types = {f.event_type for f in events[0].behavior_flags}
    assert "elevated_negative_affect" in flag_types
    assert "disengagement_signal" in flag_types


def test_on_tracker_dropped_flushes_remaining_data():
    engine = FusionEngine(window_seconds=100.0)  # long window, wouldn't auto-flush
    sid = uuid4()
    engine.observe(1, sid, 0.9, "s1", emotions={"neutral": 1.0})
    event = engine.on_tracker_dropped(1)
    assert event is not None
    assert event.student_id == sid
    assert engine.on_tracker_dropped(1) is None  # buffer already removed


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))