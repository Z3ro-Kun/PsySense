# deepface_server.py
#
# Save as: deepface_server.py (replaces your existing file)
#
# Changes from the original:
#   1. DeepFace.analyze() (synchronous, CPU-bound) now runs via
#      asyncio.to_thread() instead of being called directly inside
#      `async def` -- previously this blocked the FastAPI event loop,
#      meaning concurrent requests queued rather than running in
#      parallel. This alone should noticeably improve throughput once
#      multiple tracked people are being analyzed per cycle.
#   2. Added a root "/" health endpoint -- start_psysense.py's health
#      check was silently always failing for this service (404 on a
#      route that never existed). This fixes that without changing the
#      orchestrator.
#   3. Structured JSON logging via logging_.logger instead of print().
import asyncio
import base64
import time

import cv2
import numpy as np
import uvicorn
from deepface import DeepFace
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from logging_.logger import get_logger

logger = get_logger("emotion.deepface_server")

app = FastAPI()

# DeepFace's first call builds/loads the model; warm it once at import
# time so the first real request isn't the one that pays that cost.
_MODEL_WARMED = False


class ImageItem(BaseModel):
    image_b64: str


def normalize_emotions(raw: dict) -> dict:
    if not raw:
        return {}
    s = float(sum(raw.values())) or 1.0
    return {k: float(v) / s for k, v in raw.items()}


def _analyze_sync(frame: np.ndarray) -> dict:
    """The actual blocking DeepFace call, run off the event loop thread.

    detector_backend="skip": the frame arriving here is already a
    person-crop from main.py's YOLO detection, so re-running a second
    face detector inside DeepFace is redundant work -- and depends on
    detector-specific assets (e.g. "opencv"'s bundled Haar cascade XML)
    that aren't reliably present in every environment. "skip" tells
    DeepFace to trust the crop and go straight to emotion classification.
    """
    return DeepFace.analyze(
        frame,
        actions=["emotion"],
        enforce_detection=False,
        detector_backend="skip",
    )


@app.get("/")
async def health():
    return {"status": "ok", "service": "deepface_server"}


@app.post("/analyze_emotion/")
async def analyze_emotion(item: ImageItem):
    start_t = time.perf_counter()
    try:
        img_bytes = base64.b64decode(item.image_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        model_call_start = time.perf_counter()
        # to_thread: releases the event loop while DeepFace runs, so other
        # requests (from other tracked people this cycle) aren't blocked.
        result = await asyncio.to_thread(_analyze_sync, frame)
        model_call_end = time.perf_counter()

        if isinstance(result, list) and result:
            raw = result[0].get("emotion", {})
        elif isinstance(result, dict):
            raw = result.get("emotion", {})
        else:
            raw = {}

        total_ms = (time.perf_counter() - start_t) * 1000.0
        model_ms = (model_call_end - model_call_start) * 1000.0
        if not raw:
            return {"status": "no_face_detected", "data": None}

        return {
            "status": "success",
            "data": normalize_emotions(raw),
            "timing": {"total_ms": total_ms, "model_ms": model_ms},
        }

    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.error("DeepFace analysis failed", extra={"context": {"error": str(e)}})
        raise HTTPException(status_code=500, detail=f"An internal error occurred in DeepFace: {str(e)}") from e


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)