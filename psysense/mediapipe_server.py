# mediapipe_server.py
#
# Save as: mediapipe_server.py (replaces your existing file)
#
# Changes from the original:
#   1. Added root "/" health endpoint (start_psysense.py's health check
#      was always failing against this service before -- 404 on a route
#      that never existed).
#   2. LANDMARKER.detect() and POSE_ANALYZER.analyze_pose_cues() (both
#      blocking, CPU-bound) now run via asyncio.to_thread() instead of
#      directly inside `async def`, so concurrent requests no longer
#      serialize behind the event loop.
#   3. PoseItem gained an optional `student_id` field. When provided,
#      PoseAnalyzer keys its rolling history on that persistent identity
#      instead of the raw track_id, so tracker-ID churn/reuse can't
#      bleed one person's posture history into another's (see
#      pose_analyzer.py's docstring for the full rationale).
#   4. A background task calls POSE_ANALYZER.cleanup() once a minute to
#      evict stale per-track/per-identity history -- fixes the unbounded
#      memory growth in the original.
#   5. print() replaced with structured logging via logging_.logger.
import asyncio
import base64
import time
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pose_analyzer import PoseAnalyzer
from logging_.logger import get_logger

logger = get_logger("pose.mediapipe_server")

app = FastAPI(title="PsySense MediaPipe Pose Server", version="2.0.0")

MODEL_PATH = "pose_landmarker_heavy.task"


class PoseItem(BaseModel):
    image_b64: str
    track_id: int
    student_id: Optional[str] = None  # persistent identity key, if already resolved
    debug: Optional[bool] = False


LANDMARKER = None
try:
    base_opts = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_opts,
        output_segmentation_masks=False,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    LANDMARKER = vision.PoseLandmarker.create_from_options(options)
    logger.info(f"PoseLandmarker loaded from: {MODEL_PATH}")
except Exception as e:  # noqa: BLE001
    logger.error(f"Failed to load PoseLandmarker model: {e}")
    LANDMARKER = None

POSE_ANALYZER = PoseAnalyzer(fps=30, history_len_seconds=2.0, history_ttl_seconds=120.0)

KEY_INDICES = {
    "nose": 0, "left_shoulder": 11, "right_shoulder": 12,
    "left_hip": 23, "right_hip": 24, "left_wrist": 15, "right_wrist": 16,
}


def decode_b64_to_bgr(image_b64: str) -> Optional[np.ndarray]:
    if not image_b64:
        return None
    try:
        arr = np.frombuffer(base64.b64decode(image_b64), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def extract_landmarks_safe(detection_result) -> Optional[Any]:
    try:
        if hasattr(detection_result, "pose_landmarks") and detection_result.pose_landmarks:
            if isinstance(detection_result.pose_landmarks, (list, tuple)):
                return detection_result.pose_landmarks[0]
            return detection_result.pose_landmarks
    except Exception:
        pass
    try:
        if hasattr(detection_result, "pose_world_landmarks") and detection_result.pose_world_landmarks:
            if isinstance(detection_result.pose_world_landmarks, (list, tuple)):
                return detection_result.pose_world_landmarks[0]
            return detection_result.pose_world_landmarks
    except Exception:
        pass
    return None


def landmark_to_dict(lm) -> Dict[str, Optional[float]]:
    try:
        return {
            "x": float(getattr(lm, "x", None)) if getattr(lm, "x", None) is not None else None,
            "y": float(getattr(lm, "y", None)) if getattr(lm, "y", None) is not None else None,
            "z": float(getattr(lm, "z", None)) if getattr(lm, "z", None) is not None else None,
        }
    except Exception:
        try:
            if isinstance(lm, (list, tuple)) and len(lm) >= 2:
                return {"x": float(lm[0]), "y": float(lm[1]), "z": float(lm[2]) if len(lm) > 2 else None}
        except Exception:
            pass
    return {"x": None, "y": None, "z": None}


def _detect_sync(mp_image):
    return LANDMARKER.detect(mp_image)


def _analyze_sync(track_id: int, landmarks_obj, key: Optional[str]):
    return POSE_ANALYZER.analyze_pose_cues(track_id, landmarks_obj, key=key)


@app.get("/")
async def health():
    return {"status": "ok" if LANDMARKER is not None else "degraded", "service": "mediapipe_server"}


@app.on_event("startup")
async def _start_cleanup_loop():
    async def loop():
        while True:
            await asyncio.sleep(60.0)
            removed = POSE_ANALYZER.cleanup()
            if removed:
                logger.info(f"Pose history cleanup evicted {removed} stale entries")
    asyncio.create_task(loop())


@app.post("/analyze_pose/")
async def analyze_pose(item: PoseItem):
    if LANDMARKER is None:
        raise HTTPException(status_code=503, detail="Pose landmarker not initialized. Check model path and server logs.")

    start_total = time.perf_counter()
    try:
        img_bgr = decode_b64_to_bgr(item.image_b64)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Could not decode image_b64 to image.")

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        detect_start = time.perf_counter()
        detection_result = await asyncio.to_thread(_detect_sync, mp_image)
        detect_end = time.perf_counter()

        landmarks_obj = extract_landmarks_safe(detection_result)
        if landmarks_obj is None:
            total_ms = (time.perf_counter() - start_total) * 1000.0
            return {"status": "no_pose_detected", "data": None,
                    "timing": {"total_ms": total_ms, "landmarker_ms": (detect_end - detect_start) * 1000.0}}

        analyze_start = time.perf_counter()
        try:
            scores = await asyncio.to_thread(_analyze_sync, int(item.track_id), landmarks_obj, item.student_id)
        except Exception as e:  # noqa: BLE001
            scores = {"slumped_score": 0.0, "rigidity_score": 0.0, "fidgeting_score": 0.0}
            logger.error(f"PoseAnalyzer.analyze_pose_cues failed for track_id {item.track_id}: {e}")
        analyze_end = time.perf_counter()

        total_ms = (time.perf_counter() - start_total) * 1000.0

        debug_landmarks = None
        if item.debug:
            debug_landmarks = {}
            try:
                lm_list = landmarks_obj.landmark if hasattr(landmarks_obj, "landmark") else list(landmarks_obj)
            except Exception:
                lm_list = None
            if lm_list:
                for name, idx in KEY_INDICES.items():
                    debug_landmarks[name] = landmark_to_dict(lm_list[idx]) if idx < len(lm_list) else {"x": None, "y": None, "z": None}
            else:
                for name, idx in KEY_INDICES.items():
                    try:
                        debug_landmarks[name] = landmark_to_dict(landmarks_obj[idx])
                    except Exception:
                        debug_landmarks[name] = {"x": None, "y": None, "z": None}

        resp = {
            "status": "success",
            "data": scores,
            "timing": {
                "total_ms": float(total_ms),
                "landmarker_ms": float((detect_end - detect_start) * 1000.0),
                "analyze_ms": float((analyze_end - analyze_start) * 1000.0),
            },
        }
        if debug_landmarks is not None:
            resp["debug_landmarks"] = debug_landmarks
        return resp

    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error processing pose for track_id {getattr(item, 'track_id', 'unknown')}: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred in MediaPipe: {str(e)}") from e


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting PsySense MediaPipe server on http://0.0.0.0:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)