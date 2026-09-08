"""
core/config.py

Save as: core/config.py

Typed loader for config.yaml. Every tunable constant used by main.py (and,
going forward, anything else) is declared here with a Pydantic model, so a
typo or missing key in config.yaml fails loudly at startup with a clear
error instead of surfacing as a mysterious KeyError three layers deep
during a live camera session.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class EmbedderConfig(BaseModel):
    model_name: str = "buffalo_l"
    ctx_id: int = -1
    det_size: tuple[int, int] = (320, 320)


class QualityConfig(BaseModel):
    blur_threshold: float = 60.0
    brightness_min: float = 40.0
    brightness_max: float = 220.0
    max_yaw_deg: float = 35.0
    max_occlusion: float = 0.35
    min_face_size_px: int = 60


class MatchingConfig(BaseModel):
    match_distance_threshold: float = 0.42
    duplicate_distance_threshold: float = 0.08
    max_embeddings_per_student: int = 20


class IdentityCacheConfig(BaseModel):
    ttl_seconds: float = 30.0
    reverify_every_n_frames: int = 150


class VectorIndexConfig(BaseModel):
    backend: str = "auto"
    dim: int = 512
    persist_path: str = "data/faiss_index/students.index"


class IdentityConfig(BaseModel):
    embedder: EmbedderConfig = EmbedderConfig()
    quality: QualityConfig = QualityConfig()
    matching: MatchingConfig = MatchingConfig()
    cache: IdentityCacheConfig = IdentityCacheConfig()
    vector_index: VectorIndexConfig = VectorIndexConfig()


class CameraConfig(BaseModel):
    source: str | int = 0
    camera_id: str = "default"
    reconnect_delay_seconds: float = 3.0
    max_consecutive_read_failures: int = 30


class DetectionConfig(BaseModel):
    yolo_weights: str = "yolov8n.pt"
    person_class_id: int = 0
    conf_threshold: float = 0.25


class ByteTrackConfig(BaseModel):
    track_thresh: float = 0.5
    match_thresh: float = 0.8
    track_buffer: int = 30
    frame_rate: int = 30


class StrongSortConfig(BaseModel):
    reid_weights: str = "osnet_x0_25_msmt17.pt"
    device: str = "cpu"
    half: bool = False


class TrackerConfig(BaseModel):
    kind: str = "bytetrack"
    bytetrack: ByteTrackConfig = ByteTrackConfig()
    strongsort: StrongSortConfig = StrongSortConfig()


class FrequenciesConfig(BaseModel):
    identity_resolve_interval_sec: float = 1.0
    emotion_interval_sec: float = 2.0
    pose_interval_sec: float = 0.5
    fusion_flush_interval_sec: float = 10.0
    cache_sweep_interval_sec: float = 5.0
    pose_history_cleanup_interval_sec: float = 60.0


class ServicesConfig(BaseModel):
    emotion_url: str = "http://localhost:8000/analyze_emotion/"
    pose_url: str = "http://localhost:8001/analyze_pose/"
    request_timeout_sec: float = 1.5


class DatabaseConfig(BaseModel):
    path: str = "data/psysense.db"


class DisplayConfig(BaseModel):
    headless: bool = True


class PipelineConfig(BaseModel):
    camera: CameraConfig = CameraConfig()
    detection: DetectionConfig = DetectionConfig()
    tracker: TrackerConfig = TrackerConfig()
    frequencies: FrequenciesConfig = FrequenciesConfig()
    services: ServicesConfig = ServicesConfig()
    database: DatabaseConfig = DatabaseConfig()
    display: DisplayConfig = DisplayConfig()


class AppConfig(BaseModel):
    identity: IdentityConfig = IdentityConfig()
    pipeline: PipelineConfig = PipelineConfig()


def load_config(path: Optional[str] = None) -> AppConfig:
    config_path = Path(path) if path else Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {config_path}. PsySense does not run with hardcoded "
            f"defaults for production settings -- create config/config.yaml (see the version "
            f"shipped with the repo for the full schema)."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig(**raw)