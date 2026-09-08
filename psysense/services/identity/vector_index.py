"""
services/identity/vector_index.py

Vector similarity search for face embeddings. FAISS is the primary
backend; if FAISS is unavailable (import fails, or fails to initialize
on this platform) we transparently fall back to a linear numpy scan with
the exact same VectorIndex interface, so IdentityManager never needs to
know which one is active.

Distance metric: cosine distance, implemented as (1 - inner product) on
L2-normalized vectors so FAISS's IndexFlatIP can be used directly.
"""
from __future__ import annotations

import logging
import os
import pickle
from uuid import UUID

import numpy as np

from core.interfaces import VectorIndex
from core.exceptions import VectorIndexError

logger = logging.getLogger("psysense.identity.vector_index")


def build_vector_index(dim: int = 512, prefer_faiss: bool = True) -> VectorIndex:
    """Factory: returns a FaissVectorIndex if FAISS is importable and usable,
    otherwise a LinearVectorIndex. This is the single place fallback logic
    lives — callers just get a working VectorIndex either way."""
    if prefer_faiss:
        try:
            import faiss  # noqa: F401
            return FaissVectorIndex(dim=dim)
        except Exception as exc:  # noqa: BLE001
            logger.warning("FAISS unavailable (%s); falling back to linear vector search.", exc)
    return LinearVectorIndex(dim=dim)


class _BaseVectorIndex(VectorIndex):
    """Shared bookkeeping: maps a contiguous internal row index to
    (student_id, embedding_id), since neither FAISS's flat index nor our
    linear array natively stores arbitrary UUID metadata."""

    def __init__(self, dim: int):
        self.dim = dim
        self._row_to_ids: list[tuple[UUID, UUID]] = []

    def _record_row(self, student_id: UUID, embedding_id: UUID) -> int:
        self._row_to_ids.append((student_id, embedding_id))
        return len(self._row_to_ids) - 1

    def size(self) -> int:
        return len(self._row_to_ids)


class LinearVectorIndex(_BaseVectorIndex):
    """Brute-force cosine search via numpy. O(n) per query — fine up to a
    few thousand embeddings (typical single-school enrollment scale) and
    guarantees the system keeps functioning if FAISS can't be installed."""

    def __init__(self, dim: int = 512):
        super().__init__(dim)
        self._matrix = np.zeros((0, dim), dtype=np.float32)

    def add(self, student_id: UUID, embedding_id: UUID, vector: np.ndarray) -> None:
        vector = self._as_row(vector)
        self._matrix = np.vstack([self._matrix, vector])
        self._record_row(student_id, embedding_id)

    def remove_student(self, student_id: UUID) -> None:
        keep_rows = [i for i, (sid, _) in enumerate(self._row_to_ids) if sid != student_id]
        self._matrix = self._matrix[keep_rows] if keep_rows else np.zeros((0, self.dim), dtype=np.float32)
        self._row_to_ids = [self._row_to_ids[i] for i in keep_rows]

    def search(self, vector: np.ndarray, top_k: int = 1) -> list[tuple[UUID, UUID, float]]:
        if self.size() == 0:
            return []
        vector = self._as_row(vector)[0]
        sims = self._matrix @ vector  # cosine similarity (vectors are L2-normalized)
        distances = 1.0 - sims
        order = np.argsort(distances)[:top_k]
        return [(self._row_to_ids[i][0], self._row_to_ids[i][1], float(distances[i])) for i in order]

    def save(self, path: str) -> None:
        try:
            with open(path, "wb") as f:
                pickle.dump({"matrix": self._matrix, "row_to_ids": self._row_to_ids}, f)
        except OSError as exc:
            raise VectorIndexError(f"Failed to save linear index to {path}: {exc}") from exc

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            self._matrix = state["matrix"]
            self._row_to_ids = state["row_to_ids"]
        except (OSError, KeyError, pickle.PickleError) as exc:
            raise VectorIndexError(f"Failed to load linear index from {path}: {exc}") from exc

    @staticmethod
    def _as_row(vector: np.ndarray) -> np.ndarray:
        v = np.asarray(vector, dtype=np.float32)
        return v.reshape(1, -1) if v.ndim == 1 else v


class FaissVectorIndex(_BaseVectorIndex):
    """FAISS IndexFlatIP over L2-normalized vectors (inner product == cosine
    similarity). Deletions in FAISS's flat index aren't supported natively,
    so remove_student() rebuilds the index from surviving vectors — cheap
    at enrollment scale, and infrequent (only on student removal)."""

    def __init__(self, dim: int = 512):
        super().__init__(dim)
        import faiss
        self._faiss = faiss
        self._index = faiss.IndexFlatIP(dim)
        self._vectors: list[np.ndarray] = []  # kept only to support rebuild-on-remove

    def add(self, student_id: UUID, embedding_id: UUID, vector: np.ndarray) -> None:
        v = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        try:
            self._index.add(v)
        except Exception as exc:  # noqa: BLE001
            raise VectorIndexError(f"FAISS add() failed: {exc}") from exc
        self._vectors.append(v[0])
        self._record_row(student_id, embedding_id)

    def remove_student(self, student_id: UUID) -> None:
        keep = [(i, ids) for i, ids in enumerate(self._row_to_ids) if ids[0] != student_id]
        self._row_to_ids = [ids for _, ids in keep]
        self._vectors = [self._vectors[i] for i, _ in keep]
        self._index = self._faiss.IndexFlatIP(self.dim)
        if self._vectors:
            self._index.add(np.vstack(self._vectors).astype(np.float32))

    def search(self, vector: np.ndarray, top_k: int = 1) -> list[tuple[UUID, UUID, float]]:
        if self.size() == 0:
            return []
        v = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        try:
            sims, idxs = self._index.search(v, min(top_k, self.size()))
        except Exception as exc:  # noqa: BLE001
            raise VectorIndexError(f"FAISS search() failed: {exc}") from exc
        results = []
        for sim, idx in zip(sims[0], idxs[0]):
            if idx < 0:
                continue
            sid, eid = self._row_to_ids[idx]
            results.append((sid, eid, float(1.0 - sim)))
        return results

    def save(self, path: str) -> None:
        try:
            self._faiss.write_index(self._index, path)
            with open(path + ".meta", "wb") as f:
                pickle.dump(self._row_to_ids, f)
        except Exception as exc:  # noqa: BLE001
            raise VectorIndexError(f"Failed to save FAISS index to {path}: {exc}") from exc

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        try:
            self._index = self._faiss.read_index(path)
            with open(path + ".meta", "rb") as f:
                self._row_to_ids = pickle.load(f)
            self._vectors = [self._index.reconstruct(i) for i in range(self._index.ntotal)]
        except Exception as exc:  # noqa: BLE001
            raise VectorIndexError(f"Failed to load FAISS index from {path}: {exc}") from exc