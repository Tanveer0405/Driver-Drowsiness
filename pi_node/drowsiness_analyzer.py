"""Temporal analysis: closure, PERCLOS, fixed stare, blink rate, droop, head nod.

Rules are relative to a per-driver baseline measured during calibration, so
no hard-coded 'magic' gaze/EAR numbers are needed.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from common.config import DetectionConfig
from common.types import AlertLevel, AnalysisResult, EyeObservation
from pi_node.rolling import RollingWindow


@dataclass
class Baseline:
    ear: float
    pitch: float
    gaze_spread: float
    blink_rate: float      # blinks / minute


class DrowsinessAnalyzer:
    def __init__(self, cfg: DetectionConfig):
        self.cfg = cfg
        self.reset()

    # ---------- lifecycle ----------
    def reset(self) -> None:
        c = self.cfg
        self._t0: Optional[float] = None
        self._baseline: Optional[Baseline] = None
        self._closed_w = RollingWindow(c.perclos_window_s)
        self._ear_w = RollingWindow(c.droop_window_s)
        self._gaze_w = RollingWindow(c.stare_window_s)
        self._pitch_w = RollingWindow(c.head_window_s)
        self._blinks: deque = deque()
        self._closed_since: Optional[float] = None
        self._no_face_since: Optional[float] = None
        self._cal_ears: List[float] = []
        self._cal_pitches: List[float] = []
        self._cal_spreads: List[float] = []
        self._cal_blinks = 0
        self._last_spread_t = float("-inf")

    @property
    def calibrated(self) -> bool:
        return self._baseline is not None

    # ---------- main entry ----------
    def update(self, obs: EyeObservation, p_closed: Optional[float], now: float) -> AnalysisResult:
        if self._t0 is None:
            self._t0 = now
        if not obs.face_found:
            return self._on_no_face(now)
        self._no_face_since = None

        closed = self._is_closed(obs.ear, p_closed)
        self._track_closure(closed, now)
        self._pitch_w.add(now, obs.head_pitch)
        if not closed:                      # iris/EAR are unreliable while lids are shut
            self._gaze_w.add(now, (obs.gaze_x, obs.gaze_y))
            self._ear_w.add(now, obs.ear)

        if not self.calibrated:
            self._collect_calibration(obs, closed, now)
        return self._evaluate(obs, now)

    # ---------- closure ----------
    def _is_closed(self, ear: float, p_closed: Optional[float]) -> bool:
        ref = self._baseline.ear if self._baseline else self.cfg.default_open_ear
        ear_closed = ear < self.cfg.ear_closed_ratio * ref
        if p_closed is None:
            return ear_closed
        w = self.cfg.classifier_weight
        return (w * p_closed + (1.0 - w) * float(ear_closed)) >= 0.5

    def _track_closure(self, closed: bool, now: float) -> None:
        c = self.cfg
        if closed:
            if self._closed_since is None:
                self._closed_since = now
        elif self._closed_since is not None:
            if c.blink_min_s <= now - self._closed_since <= c.blink_max_s:
                self._blinks.append(now)
                if not self.calibrated:
                    self._cal_blinks += 1
            self._closed_since = None
        self._closed_w.add(now, float(closed))
        while self._blinks and now - self._blinks[0] > c.blink_window_s:
            self._blinks.popleft()

    # ---------- calibration ----------
    def _collect_calibration(self, obs: EyeObservation, closed: bool, now: float) -> None:
        c = self.cfg
        if not closed:
            self._cal_ears.append(obs.ear)
        self._cal_pitches.append(obs.head_pitch)
        if now - self._last_spread_t >= 1.0 and self._gaze_w.covers(now):
            self._cal_spreads.append(self._gaze_spread())
            self._last_spread_t = now

        if (now - self._t0 >= c.calibration_s
                and len(self._cal_ears) >= c.min_calibration_samples):
            minutes = c.calibration_s / 60.0
            self._baseline = Baseline(
                ear=float(np.median(self._cal_ears)),
                pitch=float(np.median(self._cal_pitches)),
                gaze_spread=float(np.median(self._cal_spreads)) if self._cal_spreads
                else c.min_gaze_std / c.stare_ratio,
                blink_rate=self._cal_blinks / minutes,
            )

    # ---------- features ----------
    def _gaze_spread(self) -> float:
        std = np.std(self._gaze_w.values(), axis=0)
        return float(np.sqrt(np.sum(std ** 2)))

    def _perclos(self, now: float) -> Optional[float]:
        if not self._closed_w.covers(now, 0.8):
            return None
        return self._closed_w.mean()

    def _blink_rate(self, now: float) -> Optional[float]:
        c = self.cfg
        if now - self._t0 < c.blink_window_s:
            return None
        return len(self._blinks) * 60.0 / c.blink_window_s

    # ---------- warning rules ----------
    def _is_stare(self, now: float) -> bool:
        c = self.cfg
        if not self._gaze_w.covers(now, 0.9):
            return False
        if any(now - b <= c.stare_window_s for b in self._blinks):
            return False
        limit = max(c.stare_ratio * self._baseline.gaze_spread, c.min_gaze_std)
        return self._gaze_spread() < limit

    def _is_low_blink_droop(self, now: float, rate: Optional[float]) -> bool:
        c = self.cfg
        if rate is None or not self._ear_w.covers(now, 0.5):
            return False
        low_rate = rate < max(c.low_blink_floor, c.low_blink_ratio * self._baseline.blink_rate)
        droop = self._ear_w.median() < c.droop_ratio * self._baseline.ear
        return low_rate and droop

    def _is_head_nod(self, now: float) -> bool:
        if not self._pitch_w.covers(now, 0.9):
            return False
        drift = abs(self._pitch_w.median() - self._baseline.pitch)
        return drift > self.cfg.head_pitch_delta

    # ---------- decision ----------
    def _evaluate(self, obs: EyeObservation, now: float) -> AnalysisResult:
        c = self.cfg
        closed_dur = (now - self._closed_since) if self._closed_since is not None else 0.0
        perclos = self._perclos(now)
        rate = self._blink_rate(now)

        reasons: List[str] = []
        alarm = False
        if closed_dur >= c.closed_alarm_s:
            alarm = True
            reasons.append("EYES_CLOSED")
        if perclos is not None and perclos >= c.perclos_threshold:
            alarm = True
            reasons.append("PERCLOS")

        warnings: List[str] = []
        if self.calibrated:
            if self._is_stare(now):
                warnings.append("STARE")
            if self._is_low_blink_droop(now, rate):
                warnings.append("LOW_BLINK_DROOP")
            if self._is_head_nod(now):
                warnings.append("HEAD_NOD")
        reasons += warnings
        if len(warnings) >= 2:
            alarm = True

        level = AlertLevel.ALARM if alarm else AlertLevel.WARNING if warnings else AlertLevel.OK
        progress = 1.0 if self.calibrated else min((now - self._t0) / c.calibration_s, 0.99)
        return AnalysisResult(
            level=level, reasons=reasons, face_found=True, ear=obs.ear,
            closed_duration=closed_dur, perclos=perclos, blink_rate=rate,
            calibrating=not self.calibrated, calibration_progress=progress,
        )

    def _on_no_face(self, now: float) -> AnalysisResult:
        if self._no_face_since is None:
            self._no_face_since = now
        self._closed_since = None
        gone = now - self._no_face_since >= self.cfg.no_face_alarm_s
        return AnalysisResult(
            level=AlertLevel.ALARM if gone else AlertLevel.OK,
            reasons=["NO_FACE"] if gone else [],
            face_found=False,
            calibrating=not self.calibrated,
            calibration_progress=1.0 if self.calibrated
            else min((now - self._t0) / self.cfg.calibration_s, 0.99),
        )
