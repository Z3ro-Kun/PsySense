"""core/exceptions.py — typed exceptions so callers can handle failures
by category instead of catching bare Exception everywhere."""


class PsySenseError(Exception):
    """Base class for all PsySense domain errors."""


class FaceQualityRejected(PsySenseError):
    """Raised (optionally) when a face fails quality gating and the caller
    requested strict mode rather than a soft MatchStatus.REJECTED_QUALITY."""


class EmbeddingExtractionError(PsySenseError):
    """The embedder failed to produce a vector (model error, corrupt image, etc.)."""


class VectorIndexError(PsySenseError):
    """FAISS / linear index failed to add, search, save, or load."""


class DuplicateEnrollmentError(PsySenseError):
    """Enrollment image is a near-duplicate of an existing embedding for the
    same student (below duplicate_distance_threshold) and was skipped."""


class DatabaseWriteError(PsySenseError):
    """Raised by the async DB writer on unrecoverable write failure."""