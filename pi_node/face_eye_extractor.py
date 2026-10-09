"""Vision front-end: frame -> eye crops, EAR, gaze ratios, head-pitch proxy."""
from __future__ import annotations

from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np

from common.types import EyeObservation


class FaceEyeExtractor:
    # MediaPipe Face Mesh landmark indices. Eye order is p1..p6 for EAR:
    # [outer corner, upper1, upper2, inner corner, lower2, lower1]
    L_EYE = [33, 160, 158, 133, 153, 144]
    R_EYE = [362, 385, 387, 263, 373, 380]
    L_IRIS, R_IRIS = 468, 473                 # need refine_landmarks=True
    L_CORNERS, R_CORNERS = (33, 133), (362, 263)   # both ordered left -> right in image
    L_LIDS, R_LIDS = (159, 145), (386, 374)        # (upper, lower)
    NOSE, CHIN = 1, 152

    def __init__(self, crop_size: int = 96, crop_scale: float = 1.6):
        self.crop_size = crop_size
        self.crop_scale = crop_scale     # square crop side = eye width * scale
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def close(self) -> None:
        self._mesh.close()

    # ---------- public ----------
    def process(self, frame_bgr: np.ndarray) -> EyeObservation:
        h, w = frame_bgr.shape[:2]
        res = self._mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return EyeObservation(face_found=False)

        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([(p.x * w, p.y * h) for p in lm])      # pixel coords
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        ear = 0.5 * (self._ear(pts, self.L_EYE) + self._ear(pts, self.R_EYE))
        gx, gy = self._gaze(pts)
        return EyeObservation(
            face_found=True,
            left_crop=self._crop(gray, pts, self.L_EYE),
            right_crop=self._crop(gray, pts, self.R_EYE),
            ear=float(ear),
            gaze_x=gx,
            gaze_y=gy,
            head_pitch=self._pitch(pts),
        )

    # ---------- geometry ----------
    @staticmethod
    def _ear(pts: np.ndarray, idx: List[int]) -> float:
        p = pts[idx]
        vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
        horizontal = np.linalg.norm(p[0] - p[3])
        return vertical / (2.0 * horizontal + 1e-6)

    def _gaze(self, pts: np.ndarray):
        """Iris position inside the eye (0..1), averaged over both eyes."""
        xs, ys = [], []
        for iris, (ca, cb), (up, lo) in (
            (self.L_IRIS, self.L_CORNERS, self.L_LIDS),
            (self.R_IRIS, self.R_CORNERS, self.R_LIDS),
        ):
            xs.append((pts[iris][0] - pts[ca][0]) / (pts[cb][0] - pts[ca][0] + 1e-6))
            ys.append((pts[iris][1] - pts[up][1]) / (pts[lo][1] - pts[up][1] + 1e-6))
        return float(np.mean(xs)), float(np.mean(ys))

    def _pitch(self, pts: np.ndarray) -> float:
        """Nose position between eye-line and chin. Shifts when the head nods."""
        eye_y = 0.5 * (pts[33][1] + pts[263][1])
        return float((pts[self.NOSE][1] - eye_y) / (pts[self.CHIN][1] - eye_y + 1e-6))

    def _crop(self, gray: np.ndarray, pts: np.ndarray, idx: List[int]) -> Optional[np.ndarray]:
        eye = pts[idx]
        cx, cy = eye.mean(axis=0)
        side = (eye[:, 0].max() - eye[:, 0].min()) * self.crop_scale
        x0, x1 = int(max(cx - side / 2, 0)), int(min(cx + side / 2, gray.shape[1]))
        y0, y1 = int(max(cy - side / 2, 0)), int(min(cy + side / 2, gray.shape[0]))
        patch = gray[y0:y1, x0:x1]
        if patch.size == 0:
            return None
        return cv2.resize(patch, (self.crop_size, self.crop_size))
