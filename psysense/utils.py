# ==========================
# utils.py
# ==========================
"""
Utility helpers used by PsySense (cropping and base64 encoding for the
HTTP calls to the emotion/pose services).
"""

import cv2
import base64
import numpy as np
from typing import Optional


def crop_with_padding(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, pad: float = 0.1) -> np.ndarray:
    """Crop a region from `frame` with padding expressed as a fraction of box size.
    Coordinates supplied as ints (pixel space). Returns the cropped BGR image.
    Handles out-of-bounds and zero-size safe-guards.
    """
    h, w = frame.shape[:2]
    # ensure ints
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=frame.dtype)

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    px = int(round(bw * pad))
    py = int(round(bh * pad))

    x1p = max(0, x1 - px)
    y1p = max(0, y1 - py)
    x2p = min(w, x2 + px)
    y2p = min(h, y2 + py)

    if x2p <= x1p or y2p <= y1p:
        return np.empty((0, 0, 3), dtype=frame.dtype)

    return frame[y1p:y2p, x1p:x2p]


def encode_image_b64(image: np.ndarray, quality: int = 90) -> str:
    """Encode BGR image to a base64 JPEG string suitable for HTTP JSON payloads."""
    if image is None or image.size == 0:
        return ""
    success, buf = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return ""
    return base64.b64encode(buf.tobytes()).decode('utf-8')


def encode_image_bytes(image: np.ndarray, quality: int = 90) -> Optional[bytes]:
    """JPEG-encode to raw bytes (no base64 layer) -- for storing directly
    in a BLOB column, e.g. pending_identity_candidates.image_jpeg."""
    if image is None or image.size == 0:
        return None
    success, buf = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return None
    return buf.tobytes()


