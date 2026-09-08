"""
services/identity/quality.py

Pre-embedding face quality gate. Runs entirely on the face crop + the
5-point landmark set InsightFace's detector already returns (no second
model needed for pose-angle estimation) — cheap enough to run on every
candidate face before we pay for an ArcFace embedding.

Checks:
  - blur          : variance of Laplacian
  - illumination  : mean brightness in/out of an acceptable band
  - pose angle    : yaw estimated from eye/nose/mouth landmark geometry
  - occlusion     : landmark-symmetry heuristic (cheap proxy; can be swapped
                    for a learned occlusion model later without changing
                    the interface)
  - face size     : shorter side of the detected box, in pixels
"""
from __future__ import annotations

import cv2
import numpy as np

from core.interfaces import QualityAssessor
from core.models import FaceQualityReport, QualityRejectReason


class HeuristicQualityAssessor(QualityAssessor):
    def __init__(
        self,
        blur_threshold: float = 60.0,
        brightness_range: tuple[float, float] = (40.0, 220.0),
        max_yaw_deg: float = 35.0,
        max_occlusion: float = 0.35,
        min_face_size_px: int = 60,
    ) -> None:
        self.blur_threshold = blur_threshold
        self.brightness_low, self.brightness_high = brightness_range
        self.max_yaw_deg = max_yaw_deg
        self.max_occlusion = max_occlusion
        self.min_face_size_px = min_face_size_px

    # ------------------------------------------------------------------ #
    def assess(
        self,
        face_bgr: np.ndarray,
        face_box: tuple[int, int, int, int],
        landmarks_5pt: np.ndarray | None = None,
    ) -> FaceQualityReport:
        reasons: list[QualityRejectReason] = []

        if face_bgr is None or face_bgr.size == 0:
            return FaceQualityReport(
                passed=False, blur_score=0.0, brightness=0.0, occlusion_score=1.0,
                face_size_px=0, reasons=[QualityRejectReason.NO_FACE],
            )

        x1, y1, x2, y2 = face_box
        face_size_px = min(max(0, x2 - x1), max(0, y2 - y1))
        if face_size_px < self.min_face_size_px:
            reasons.append(QualityRejectReason.TOO_SMALL)

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_score < self.blur_threshold:
            reasons.append(QualityRejectReason.BLUR)

        brightness = float(np.mean(gray))
        if brightness < self.brightness_low:
            reasons.append(QualityRejectReason.ILLUMINATION_LOW)
        elif brightness > self.brightness_high:
            reasons.append(QualityRejectReason.ILLUMINATION_HIGH)

        yaw_deg, occlusion_score = self._pose_and_occlusion(landmarks_5pt)
        if yaw_deg is not None and abs(yaw_deg) > self.max_yaw_deg:
            reasons.append(QualityRejectReason.POSE_ANGLE)
        if occlusion_score > self.max_occlusion:
            reasons.append(QualityRejectReason.OCCLUSION)

        return FaceQualityReport(
            passed=len(reasons) == 0,
            blur_score=blur_score,
            brightness=brightness,
            yaw_deg=yaw_deg,
            pitch_deg=None,
            occlusion_score=occlusion_score,
            face_size_px=face_size_px,
            reasons=reasons,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _pose_and_occlusion(
        landmarks_5pt: np.ndarray | None,
    ) -> tuple[float | None, float]:
        """
        Estimate yaw and a cheap occlusion proxy from InsightFace's standard
        5-point landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth].

        Yaw: derived from the horizontal asymmetry of the nose relative to
        the eye midpoint, normalized by inter-eye distance.

        Occlusion proxy: landmark confidence isn't exposed by the 5-point
        set, so we use left/right symmetry breakdown as a stand-in — a
        heavily occluded half-face produces large left/right asymmetry in
        both eye-to-nose and mouth-to-nose distances. This is intentionally
        a placeholder that can be swapped for a learned occlusion classifier
        later without changing the QualityAssessor interface.
        """
        if landmarks_5pt is None or len(landmarks_5pt) < 5:
            return None, 0.0

        left_eye, right_eye, nose, left_mouth, right_mouth = landmarks_5pt[:5]

        eye_dist = float(np.linalg.norm(right_eye - left_eye)) or 1e-6
        eye_mid = (left_eye + right_eye) / 2.0
        nose_offset_x = float(nose[0] - eye_mid[0])
        yaw_deg = float(np.clip((nose_offset_x / eye_dist) * 90.0, -90.0, 90.0))

        left_span = float(np.linalg.norm(nose - left_mouth))
        right_span = float(np.linalg.norm(nose - right_mouth))
        span_sum = left_span + right_span or 1e-6
        asymmetry = abs(left_span - right_span) / span_sum
        occlusion_score = float(np.clip(asymmetry * 2.0, 0.0, 1.0))

        return yaw_deg, occlusion_score