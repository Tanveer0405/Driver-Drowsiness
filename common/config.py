"""Central configuration. Every tunable number lives here."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NetworkConfig:
    pi_host: str = "raspberrypi.local"
    port: int = 5555
    frame_width: int = 640
    frame_height: int = 480
    jpeg_quality: int = 70
    target_fps: int = 15
    reply_timeout_ms: int = 5000


@dataclass(frozen=True)
class ModelConfig:
    model_path: str = "models/eye_state.tflite"
    img_size: int = 96          # must match TrainConfig.img_size
    num_threads: int = 4


@dataclass(frozen=True)
class DetectionConfig:
    # --- per-driver calibration ---
    calibration_s: float = 45.0
    default_open_ear: float = 0.30
    min_calibration_samples: int = 30

    # --- eye closure ---
    ear_closed_ratio: float = 0.70     # EAR < ratio * baseline -> closed
    classifier_weight: float = 0.70    # CNN vs EAR weight in fusion
    closed_alarm_s: float = 1.5
    blink_min_s: float = 0.04
    blink_max_s: float = 0.80
    perclos_window_s: float = 30.0
    perclos_threshold: float = 0.30
    no_face_alarm_s: float = 3.0

    # --- fixed stare ---
    stare_window_s: float = 10.0
    stare_ratio: float = 0.30          # spread < ratio * baseline spread
    min_gaze_std: float = 0.004

    # --- low blink rate + eyelid droop ---
    blink_window_s: float = 60.0
    low_blink_ratio: float = 0.30
    low_blink_floor: float = 2.0       # blinks / minute
    droop_ratio: float = 0.80
    droop_window_s: float = 10.0

    # --- head nodding ---
    head_window_s: float = 3.0
    head_pitch_delta: float = 0.12


@dataclass(frozen=True)
class TrainConfig:
    data_dir: str = "data/eyes"        # contains open/ and closed/
    out_dir: str = "models"
    img_size: int = 96
    batch_size: int = 64
    alpha: float = 0.35
    val_split: float = 0.15
    epochs_head: int = 6
    epochs_finetune: int = 6
    lr_head: float = 1e-3
    lr_finetune: float = 1e-5
    unfreeze_layers: int = 40
    quantization: str = "dynamic"      # "dynamic" | "float16" | "none"
    seed: int = 42
