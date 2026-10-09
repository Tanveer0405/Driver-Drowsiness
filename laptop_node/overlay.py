"""Draws status text on the preview window."""
from __future__ import annotations

import cv2

from common.types import AlertLevel

COLORS = {AlertLevel.OK: (0, 200, 0), AlertLevel.WARNING: (0, 200, 255), AlertLevel.ALARM: (0, 0, 255)}


class StatusOverlay:
    def draw(self, frame, result, level: AlertLevel, link_ok: bool):
        color = COLORS[level]
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 8), color, -1)
        lines = []
        if not link_ok:
            lines.append("Pi not responding")
        elif result:
            if result.get("calibrating"):
                lines.append(f"Calibrating {result['calibration_progress']*100:.0f}% - look at the road, blink normally")
            lines.append(f"Level: {level.name}  {' '.join(result.get('reasons', []))}")
            lines.append(f"EAR {result.get('ear', 0):.2f}  closed {result.get('closed_duration', 0):.1f}s")
            p = result.get("perclos")
            lines.append(f"PERCLOS {p:.2f}" if p is not None else "PERCLOS --")
            if not result.get("face_found", True):
                lines.append("No face")
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (10, 32 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame
