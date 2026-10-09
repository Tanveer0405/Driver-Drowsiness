"""Time-based rolling window (uses timestamps, not frame counts)."""
from __future__ import annotations

from collections import deque

import numpy as np


class RollingWindow:
    def __init__(self, span_s: float):
        self.span = span_s
        self._d: deque = deque()   # (timestamp, value)

    def add(self, t: float, value) -> None:
        self._d.append((t, value))
        self._trim(t)

    def _trim(self, now: float) -> None:
        while self._d and now - self._d[0][0] > self.span:
            self._d.popleft()

    def __len__(self) -> int:
        return len(self._d)

    def covers(self, now: float, fraction: float = 0.9) -> bool:
        """True when the window holds at least `fraction` of its span of data."""
        self._trim(now)
        return bool(self._d) and (now - self._d[0][0]) >= self.span * fraction

    def values(self) -> np.ndarray:
        return np.array([v for _, v in self._d], dtype=np.float64)

    def mean(self) -> float:
        return float(np.mean(self.values()))

    def median(self) -> float:
        return float(np.median(self.values()))
