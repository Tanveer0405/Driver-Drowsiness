"""Test the whole pipeline on one machine (e.g. your Mac) with its webcam.
    python local_demo.py --model models/eye_state.tflite"""
from __future__ import annotations

import argparse
import time

import cv2

from common.config import DetectionConfig, ModelConfig
from common.types import AlertLevel
from laptop_node.overlay import StatusOverlay
from pi_node.drowsiness_analyzer import DrowsinessAnalyzer
from pi_node.eye_classifier import EyeStateClassifier
from pi_node.face_eye_extractor import FaceEyeExtractor

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=ModelConfig.model_path)
ap.add_argument("--camera", type=int, default=0)
a = ap.parse_args()

mcfg = ModelConfig(model_path=a.model)
extractor, clf = FaceEyeExtractor(mcfg.img_size), EyeStateClassifier(mcfg)
analyzer, overlay = DrowsinessAnalyzer(DetectionConfig()), StatusOverlay()
cap = cv2.VideoCapture(a.camera)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    obs = extractor.process(frame)
    p = clf.predict_closed_prob([obs.left_crop, obs.right_crop]) if obs.face_found else None
    res = analyzer.update(obs, p, time.monotonic()).to_dict()
    cv2.imshow("local demo", overlay.draw(cv2.flip(frame, 1), res, AlertLevel(res["level"]), True))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
