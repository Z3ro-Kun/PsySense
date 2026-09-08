"""
services/tracking/tracker.py

Save as: services/tracking/tracker.py

Common interface over ByteTrack and StrongSort (both available via the
boxmot package, already a dependency from the original codebase's
StrongSort usage -- no new heavy dependency needed to benchmark
ByteTrack). Both backends expose the same .update(dets, frame) -> tracks
contract, so main.py and the benchmark script don't need to know which
one is active.

Track ID stability from the tracker is explicitly NOT relied upon for
identity anywhere in this system (see services/identity/manager.py) --
this wrapper's only job is short-term frame-to-frame association good
enough for the Active Identity Cache to skip repeated vector searches
within a few frames of the same person. That's what makes ByteTrack a
legitimate candidate here despite lacking StrongSort's ReID: identity
correctness no longer depends on the tracker at all.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from logging_.logger import get_logger

logger = get_logger("tracking.tracker")


class Tracker(ABC):
    @abstractmethod
    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        detections: Nx6 array [x1, y1, x2, y2, conf, cls]
        returns: Mx7+ array [x1, y1, x2, y2, track_id, conf, cls, ...]
        """

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class ByteTrackTracker(Tracker):
    def __init__(self, track_thresh: float = 0.5, match_thresh: float = 0.8, track_buffer: int = 30, frame_rate: int = 30):
        from boxmot import ByteTrack  # local import: keep boxmot optional until actually used

        self._impl = ByteTrack(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            frame_rate=frame_rate,
        )

    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        return self._impl.update(detections, frame)

    @property
    def name(self) -> str:
        return "bytetrack"


class StrongSortTracker(Tracker):
    def __init__(self, reid_weights: Optional[str] = None, device: str = "cpu", half: bool = False):
        from boxmot import StrongSort

        weights_path = Path(reid_weights) if reid_weights else None
        self._impl = StrongSort(reid_weights=weights_path, device=device, half=half)

    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        return self._impl.update(detections, frame)

    @property
    def name(self) -> str:
        return "strongsort"


def build_tracker(kind: str, **kwargs) -> Tracker:
    """Factory. kind: 'bytetrack' | 'strongsort'. Raises ValueError on
    unknown kind rather than silently defaulting -- a typo in config.yaml
    should fail loudly at startup, not fall back to the wrong tracker."""
    kind = kind.lower().strip()
    if kind == "bytetrack":
        return ByteTrackTracker(**kwargs)
    if kind == "strongsort":
        return StrongSortTracker(**kwargs)
    raise ValueError(f"Unknown tracker kind '{kind}'. Expected 'bytetrack' or 'strongsort'.")