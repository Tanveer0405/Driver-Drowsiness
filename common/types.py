"""Plain data containers passed between components."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import List, Optional

import numpy as np


class AlertLevel(IntEnum):
    OK = 0
    WARNING = 1
    ALARM = 2


@dataclass
class EyeObservation:
    """What the vision front-end extracted from one frame."""
    face_found: bool
    left_crop: Optional[np.ndarray] = None   # grayscale uint8 (S, S)
    right_crop: Optional[np.ndarray] = None
    ear: float = 0.0
    gaze_x: float = 0.5
    gaze_y: float = 0.5
    head_pitch: float = 0.0                  # nose-position ratio proxy


@dataclass
class AnalysisResult:
    """What the temporal analyzer decided. Sent back to the laptop as JSON."""
    level: AlertLevel = AlertLevel.OK
    reasons: List[str] = field(default_factory=list)
    face_found: bool = False
    ear: float = 0.0
    closed_duration: float = 0.0
    perclos: Optional[float] = None
    blink_rate: Optional[float] = None
    calibrating: bool = True
    calibration_progress: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["level"] = int(self.level)
        return d
