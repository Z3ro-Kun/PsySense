# pose_analyzer.py
#
# Save as: pose_analyzer.py (replaces your existing file)
#
# Changes from the original:
#   - self.history now tracks a last-touched timestamp per key and
#     exposes cleanup(ttl_seconds) to evict stale entries. The original
#     defaultdict(deque) never removed entries for tracks no longer seen,
#     which grows without bound over a long-running session. Call
#     cleanup() periodically (mediapipe_server.py does this once a
#     second in the patched version below).
#   - analyze_pose_cues() now accepts an optional `key` override so the
#     caller can pass a persistent student_id instead of a raw track_id
#     once identity is resolved -- keeping history keyed on a stable
#     identity rather than a tracker ID that can churn or be reused
#     avoids the cross-contamination edge case flagged in the audit.
#     Falls back to track_id if no override is given, so this is a
#     backward-compatible change.
"""
Converts MediaPipe landmarks into normalized pose cues, fused later with
emotion results by the Fusion Engine.
"""

import math
import time
from collections import deque
from typing import Optional, Union

import numpy as np

MP_NOSE = 0
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_WRIST = 15
MP_RIGHT_WRIST = 16


class PoseAnalyzer:
    def __init__(self, fps: int = 30, history_len_seconds: float = 2.0, history_ttl_seconds: float = 120.0):
        self.fps = fps
        self.history_len = max(1, int(fps * history_len_seconds))
        self.history_ttl_seconds = history_ttl_seconds
        # key -> deque of {'time', 'torso_ang', 'lw', 'rw'}
        self.history: dict[Union[int, str], deque] = {}
        # key -> last time this key was touched, for TTL eviction
        self._last_touched: dict[Union[int, str], float] = {}

    def cleanup(self, ttl_seconds: Optional[float] = None) -> int:
        """Evict history for keys not touched within ttl_seconds. Returns
        the number of keys removed. Call this periodically (e.g. once per
        second) rather than never, which was the original bug."""
        ttl = ttl_seconds if ttl_seconds is not None else self.history_ttl_seconds
        now = time.time()
        stale = [k for k, t in self._last_touched.items() if now - t > ttl]
        for k in stale:
            self.history.pop(k, None)
            self._last_touched.pop(k, None)
        return len(stale)

    def drop(self, key: Union[int, str]) -> None:
        """Explicitly remove history for a key (e.g. when the tracker
        drops an ID or identity resolution determines the key changed)."""
        self.history.pop(key, None)
        self._last_touched.pop(key, None)

    def _get_point(self, landmarks, idx):
        try:
            lm = landmarks[idx]
            x = getattr(lm, 'x', None)
            y = getattr(lm, 'y', None)
            z = getattr(lm, 'z', None)
            if x is not None and y is not None:
                return float(x), float(y), float(z or 0.0)
        except Exception:
            pass
        try:
            lm = landmarks[idx]
            if isinstance(lm, (list, tuple, np.ndarray)) and len(lm) >= 2:
                x, y = float(lm[0]), float(lm[1])
                z = float(lm[2]) if len(lm) > 2 else 0.0
                return x, y, z
        except Exception:
            pass
        return None

    def _vector(self, a, b):
        return (b[0] - a[0], b[1] - a[1])

    def analyze_pose_cues(self, track_id: int, landmarks, key: Optional[Union[int, str]] = None) -> dict:
        """
        key: optional persistent identity key (e.g. str(student_id)) to
        use for history instead of the raw tracker ID. Defaults to
        track_id for backward compatibility with existing callers.
        """
        history_key = key if key is not None else track_id

        if landmarks is None:
            return {"slumped_score": 0.0, "rigidity_score": 0.0, "fidgeting_score": 0.0}

        nose = self._get_point(landmarks, MP_NOSE)
        ls = self._get_point(landmarks, MP_LEFT_SHOULDER)
        rs = self._get_point(landmarks, MP_RIGHT_SHOULDER)
        lh = self._get_point(landmarks, MP_LEFT_HIP)
        rh = self._get_point(landmarks, MP_RIGHT_HIP)
        lw = self._get_point(landmarks, MP_LEFT_WRIST)
        rw = self._get_point(landmarks, MP_RIGHT_WRIST)

        if not (nose and ls and rs and lh and rh):
            return {"slumped_score": 0.0, "rigidity_score": 0.0, "fidgeting_score": 0.0}

        shoulders_y = (ls[1] + rs[1]) / 2.0
        hips_y = (lh[1] + rh[1]) / 2.0
        torso_height = max(1e-6, abs(hips_y - shoulders_y))

        nose_to_shoulders = (nose[1] - shoulders_y) / torso_height
        slumped_raw = max(0.0, min(2.0, nose_to_shoulders))
        slumped_score = (slumped_raw / 1.0) if slumped_raw < 1.0 else 1.0

        torso_vec = self._vector(((ls[0] + rs[0]) / 2.0, shoulders_y), ((lh[0] + rh[0]) / 2.0, hips_y))
        torso_ang = math.atan2(torso_vec[1], torso_vec[0])

        if history_key not in self.history:
            self.history[history_key] = deque(maxlen=self.history_len)
        hist = self.history[history_key]
        now = time.time()
        hist.append({'time': now, 'torso_ang': torso_ang, 'lw': lw, 'rw': rw})
        self._last_touched[history_key] = now

        angs = [h['torso_ang'] for h in hist]
        ang_var = float(np.var(angs)) if len(angs) >= 2 else 0.0
        rigidity_score = 1.0 - (math.tanh(ang_var * 50.0))
        rigidity_score = float(max(0.0, min(1.0, rigidity_score)))

        wrist_moves = []
        for i in range(1, len(hist)):
            prev, cur = hist[i - 1], hist[i]
            if prev['lw'] and cur['lw']:
                wrist_moves.append(math.hypot(cur['lw'][0] - prev['lw'][0], cur['lw'][1] - prev['lw'][1]))
            if prev['rw'] and cur['rw']:
                wrist_moves.append(math.hypot(cur['rw'][0] - prev['rw'][0], cur['rw'][1] - prev['rw'][1]))
        avg_move = float(np.mean(wrist_moves)) if wrist_moves else 0.0
        fidget_raw = avg_move / max(1e-6, torso_height)
        fidget_score = float(max(0.0, min(1.0, fidget_raw * 5.0)))

        return {
            'slumped_score': float(slumped_score),
            'rigidity_score': float(rigidity_score),
            'fidgeting_score': float(fidget_score),
        }