"""
core/interfaces.py

Abstract interfaces (ports) for the identity subsystem. Concrete
implementations live in services/identity/*. Depending on these
abstractions (not concrete classes) is what lets us swap FAISS for a
linear-scan fallback, or ArcFace for another embedder, without touching
IdentityManager.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

import numpy as np

from core.models import FaceQualityReport


class QualityAssessor(ABC):
    @abstractmethod
    def assess(self, face_bgr: np.ndarray, face_box: tuple[int, int, int, int]) -> FaceQualityReport:
        """Evaluate a face crop and return a pass/fail quality report."""


class EmbeddingExtractor(ABC):
    @abstractmethod
    def embed(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return a 512-d L2-normalized embedding, or None if extraction failed."""

    @property
    @abstractmethod
    def dim(self) -> int:
        ...


class VectorIndex(ABC):
    """Backend-agnostic nearest-neighbor search over face embeddings."""

    @abstractmethod
    def add(self, student_id: UUID, embedding_id: UUID, vector: np.ndarray) -> None:
        ...

    @abstractmethod
    def remove_student(self, student_id: UUID) -> None:
        ...

    @abstractmethod
    def search(self, vector: np.ndarray, top_k: int = 1) -> list[tuple[UUID, UUID, float]]:
        """Return list of (student_id, embedding_id, distance) sorted ascending by distance."""

    @abstractmethod
    def size(self) -> int:
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...