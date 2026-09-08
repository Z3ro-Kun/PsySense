"""
benchmarks/bench_tracker.py

Save as: benchmarks/bench_tracker.py

Run on your machine to settle the ByteTrack-vs-StrongSort question with
real numbers instead of my a-priori reasoning. Feeds the same YOLO
detections, frame by frame, into both trackers and reports:
  - mean / p95 update() latency (ms)
  - effective FPS the tracker stage alone could sustain
  - ID-switch count: how many times a spatially-continuous track (same
    detection lineage, judged by IoU continuity) changed track_id --
    this quantifies exactly how much MORE the Active Identity Cache will
    have to re-verify under ByteTrack vs StrongSort, which is the real
    cost of dropping ReID now that identity itself doesn't depend on it.

Usage:
    python -m benchmarks.bench_tracker --source path/to/video.mp4
    python -m benchmarks.bench_tracker --source 0          # webcam
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from services.tracking.tracker import build_tracker
from logging_.logger import get_logger

logger = get_logger("benchmarks.tracker")


@dataclass
class TrackerBenchResult:
    name: str
    frames: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    id_switches: int = 0
    unique_ids_seen: set = field(default_factory=set)

    def summary(self) -> dict:
        arr = np.array(self.latencies_ms) if self.latencies_ms else np.array([0.0])
        return {
            "tracker": self.name,
            "frames": self.frames,
            "mean_latency_ms": float(arr.mean()),
            "p95_latency_ms": float(np.percentile(arr, 95)),
            "sustainable_fps": float(1000.0 / arr.mean()) if arr.mean() > 0 else float("inf"),
            "id_switches": self.id_switches,
            "unique_ids_seen": len(self.unique_ids_seen),
        }


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xa2, ya2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _detect_persons(yolo, frame: np.ndarray, conf_thresh: float = 0.25) -> np.ndarray:
    results = yolo(frame, verbose=False)[0]
    dets = []
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls != 0 or conf < conf_thresh:  # class 0 = person in COCO
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        dets.append([x1, y1, x2, y2, conf, cls])
    return np.array(dets, dtype=np.float32) if dets else np.empty((0, 6), dtype=np.float32)


def run_benchmark(source: str, yolo_weights: str, max_frames: int, tracker_kinds: list[str]) -> list[dict]:
    from ultralytics import YOLO

    yolo = YOLO(yolo_weights)
    cap_source = int(source) if source.isdigit() else source
    results: list[dict] = []

    for kind in tracker_kinds:
        logger.info(f"Benchmarking tracker: {kind}")
        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        tracker = build_tracker(kind)
        result = TrackerBenchResult(name=kind)

        # Track previous frame's (box, id) pairs to detect ID switches on
        # spatially-continuous detections (high IoU with previous frame
        # but a different track_id than last time in that same location).
        prev_tracks: list[tuple[np.ndarray, int]] = []

        frame_idx = 0
        while frame_idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            dets = _detect_persons(yolo, frame)

            start = time.perf_counter()
            tracks = tracker.update(dets, frame)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            result.frames += 1
            result.latencies_ms.append(elapsed_ms)

            current_tracks: list[tuple[np.ndarray, int]] = []
            for row in tracks:
                box, tid = row[:4].astype(np.float32), int(row[4])
                result.unique_ids_seen.add(tid)
                current_tracks.append((box, tid))
                # Compare against previous frame: same physical location
                # (high IoU) but different ID than we'd expect = ID switch.
                for prev_box, prev_id in prev_tracks:
                    if _iou(box, prev_box) > 0.6 and prev_id != tid:
                        result.id_switches += 1
                        break

            prev_tracks = current_tracks
            frame_idx += 1

        cap.release()
        results.append(result.summary())
        logger.info(f"Result: {result.summary()}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ByteTrack vs StrongSort on real video")
    parser.add_argument("--source", required=True, help="Video file path or webcam index (e.g. 0)")
    parser.add_argument("--yolo-weights", default="yolov8n.pt")
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--trackers", nargs="+", default=["bytetrack", "strongsort"])
    args = parser.parse_args()

    results = run_benchmark(args.source, args.yolo_weights, args.max_frames, args.trackers)

    print("\n=== Tracker Benchmark Results ===")
    print(f"{'tracker':<12} {'fps':>8} {'mean_ms':>10} {'p95_ms':>10} {'id_switches':>12} {'unique_ids':>11}")
    for r in results:
        print(f"{r['tracker']:<12} {r['sustainable_fps']:>8.1f} {r['mean_latency_ms']:>10.2f} "
              f"{r['p95_latency_ms']:>10.2f} {r['id_switches']:>12} {r['unique_ids_seen']:>11}")

    print(
        "\nInterpretation: ByteTrack should show materially lower mean_ms / higher fps.\n"
        "id_switches will likely be higher for ByteTrack -- that's expected and, per the\n"
        "architecture, tolerable: the Identity Manager re-verifies via ArcFace regardless\n"
        "of tracker ID stability, and the Active Identity Cache's periodic forced\n"
        "re-verification bounds the cost of extra switches. Only escalate back to\n"
        "StrongSort if id_switches is so high it's driving vector-search load past what\n"
        "the identity subsystem can sustain at your target FPS."
    )


if __name__ == "__main__":
    main()