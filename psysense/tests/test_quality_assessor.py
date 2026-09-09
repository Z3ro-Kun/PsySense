"""
tests/test_quality_assessor.py

Regression test for the bug found while investigating "enrolled student
still resolves as unknown": HeuristicQualityAssessor.assess() used to
compute blur/brightness over the *entire* input image regardless of
face_box, only using face_box for the size check. That meant a caller
passing something larger than the actual face (e.g. main.py's resolve()
path passing the full YOLO person-crop) got quality metrics reflecting
clothing/background, not the face -- inconsistent with enrollment-time
assessment of the same person under the same conditions. This test proves
assess() now actually uses the face_box region for blur/brightness.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models import QualityRejectReason  # noqa: E402
from services.identity.quality import HeuristicQualityAssessor  # noqa: E402


def _image_with_bright_textured_patch_on_dark_flat_background() -> np.ndarray:
    # Whole image: mostly a very dark, flat background -- the small bright
    # patch isn't enough to pull the whole-image mean brightness above the
    # threshold, so assessing the whole image fails illumination.
    img = np.full((200, 200, 3), 5, dtype=np.uint8)
    rng = np.random.default_rng(42)
    img[70:130, 70:130] = rng.integers(150, 200, size=(60, 60, 3), dtype=np.uint8)
    return img


def test_assess_uses_face_box_region_for_blur_and_brightness():
    img = _image_with_bright_textured_patch_on_dark_flat_background()
    assessor = HeuristicQualityAssessor(blur_threshold=10.0, brightness_range=(40.0, 220.0), min_face_size_px=10)

    face_only = assessor.assess(img, face_box=(70, 70, 130, 130))
    assert QualityRejectReason.ILLUMINATION_LOW not in face_only.reasons
    assert QualityRejectReason.BLUR not in face_only.reasons

    whole_image = assessor.assess(img, face_box=(0, 0, 200, 200))
    assert QualityRejectReason.ILLUMINATION_LOW in whole_image.reasons


def test_assess_handles_out_of_bounds_face_box_without_crashing():
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    assessor = HeuristicQualityAssessor(min_face_size_px=10)
    # Degenerate/out-of-bounds box -- must fall back gracefully, not raise.
    report = assessor.assess(img, face_box=(90, 90, 200, 200))
    assert report.face_size_px >= 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
