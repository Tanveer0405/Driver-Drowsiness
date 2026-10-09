"""Plays warning / alarm sounds on the laptop speaker (synthesized, no audio files)."""
from __future__ import annotations

import time

import numpy as np
import pygame

from common.types import AlertLevel


class AlarmPlayer:
    SR = 44100

    def __init__(self, hold_s: float = 2.0, warn_every_s: float = 3.0):
        pygame.mixer.init(frequency=self.SR, size=-16, channels=1)
        self._beep = self._make([(660, 0.25)], 0.35)
        self._alarm = self._make([(1000, 0.2), (700, 0.2)] * 2, 0.9)
        self._chan = None
        self._hold_s, self._warn_every = hold_s, warn_every_s
        self._alarm_until = 0.0
        self._last_warn = 0.0

    def _make(self, tones, volume: float) -> pygame.mixer.Sound:
        parts = []
        for freq, dur in tones:
            t = np.linspace(0, dur, int(self.SR * dur), endpoint=False)
            parts.append(np.sign(np.sin(2 * np.pi * freq * t)) * volume * 0.5)
        wave = np.concatenate(parts)
        return pygame.sndarray.make_sound((wave * 32767).astype(np.int16))

    def _busy(self) -> bool:
        return self._chan is not None and self._chan.get_busy()

    def update(self, level: AlertLevel) -> None:
        now = time.monotonic()
        if level == AlertLevel.ALARM:
            self._alarm_until = now + self._hold_s
        if now < self._alarm_until:
            if not self._busy():
                self._chan = self._alarm.play()
        elif level == AlertLevel.WARNING:
            if not self._busy() and now - self._last_warn >= self._warn_every:
                self._chan = self._beep.play()
                self._last_warn = now
        elif self._busy():
            self._chan.stop()
