"""
services/identity/embedder.py

Wraps InsightFace's ArcFace recognition model. Loaded once (module-level
singleton via get_embedder()) and reused across the whole process —
model construction is the expensive part, inference is cheap.

Replaces the old DeepFace-based identity_server.py / DeepFace.find()
approach with a proper embedding extractor that hands off to a
VectorIndex (FAISS or linear) rather than doing DB search itself.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from core.interfaces import EmbeddingExtractor
from core.exceptions import EmbeddingExtractionError

logger = logging.getLogger("psysense.identity.embedder")

_EMBEDDER_SINGLETON: Optional["ArcFaceEmbedder"] = None


class ArcFaceEmbedder(EmbeddingExtractor):
    """
    Thin wrapper around insightface.app.FaceAnalysis, used purely for its
    recognition ('recognition') model. Detection is assumed to already be
    done upstream (YOLO person box -> face region), but we still run
    InsightFace's own detector on the crop because we need its 5-point
    landmarks for FaceQualityAssessor's pose/occlusion estimate, and its
    ArcFace embedding for identity.
    """

    _DIM = 512

    def __init__(self, ctx_id: int = -1, det_size: tuple[int, int] = (320, 320), model_name: str = "buffalo_l"):
        """
        ctx_id: -1 = CPU, >=0 = GPU device id (auto-fallback to CPU on failure).
        """
        self._app = None
        self._ctx_id_requested = ctx_id
        self._det_size = det_size
        self._model_name = model_name
        self._load(ctx_id)

    def _load(self, ctx_id: int) -> None:
        try:
            from insightface.app import FaceAnalysis  # local import: heavy dep, load lazily

            app = FaceAnalysis(name=self._model_name, providers=self._providers_for(ctx_id))
            app.prepare(ctx_id=ctx_id, det_size=self._det_size)
            self._app = app
            logger.info(
                "ArcFace embedder loaded (model=%s, ctx_id=%s, det_size=%s)",
                self._model_name, ctx_id, self._det_size,
            )
        except Exception as exc:  # noqa: BLE001 - deliberate: fall back to CPU once, then raise
            if ctx_id != -1:
                logger.warning("GPU init failed for InsightFace (%s); falling back to CPU.", exc)
                self._load(ctx_id=-1)
            else:
                logger.error("Failed to load InsightFace ArcFace model: %s", exc)
                raise EmbeddingExtractionError(f"Could not initialize InsightFace: {exc}") from exc

    @staticmethod
    def _providers_for(ctx_id: int) -> list[str]:
        if ctx_id >= 0:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    @property
    def dim(self) -> int:
        return self._DIM

    def analyze(self, face_bgr: np.ndarray):
        """
        Run full InsightFace analysis (detect + landmark + embed) on a crop.
        Returns the InsightFace Face object for the largest detected face,
        or None. Exposed separately from embed() because quality.py needs
        the landmarks too, and we don't want to run detection twice.
        """
        if face_bgr is None or face_bgr.size == 0 or self._app is None:
            return None
        try:
            faces = self._app.get(face_bgr)
        except Exception as exc:  # noqa: BLE001
            logger.warning("InsightFace inference failed: %s", exc)
            return None
        if not faces:
            return None
        # If InsightFace's own detector finds multiple faces in the crop
        # (shouldn't normally happen since we feed it a person-cropped
        # region), keep the largest bounding box as the primary subject.
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        return faces[0]

    def embed(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        face = self.analyze(face_bgr)
        if face is None or getattr(face, "normed_embedding", None) is None:
            return None
        vec = np.asarray(face.normed_embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


def get_embedder(ctx_id: int = -1, det_size: tuple[int, int] = (320, 320)) -> ArcFaceEmbedder:
    """Process-wide singleton accessor so the model is loaded exactly once."""
    global _EMBEDDER_SINGLETON
    if _EMBEDDER_SINGLETON is None:
        _EMBEDDER_SINGLETON = ArcFaceEmbedder(ctx_id=ctx_id, det_size=det_size)
    return _EMBEDDER_SINGLETON