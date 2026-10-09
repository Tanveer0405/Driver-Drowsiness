"""Webcam capture."""
from __future__ import annotations

import cv2


class Webcam:
    def __init__(self, index: int, width: int, height: int):
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._cap.isOpened():
            raise RuntimeError("Cannot open webcam (check camera permission on macOS)")

    def read(self):
        return self._cap.read()

    def release(self) -> None:
        self._cap.release()
