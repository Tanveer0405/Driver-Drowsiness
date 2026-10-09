"""Receives JPEG frames from the laptop, replies with the analysis as JSON."""
from __future__ import annotations

import time

import cv2
import numpy as np
import zmq

from common.config import DetectionConfig, ModelConfig, NetworkConfig
from common.types import AnalysisResult
from pi_node.drowsiness_analyzer import DrowsinessAnalyzer
from pi_node.eye_classifier import EyeStateClassifier
from pi_node.face_eye_extractor import FaceEyeExtractor


class DrowsinessServer:
    RESET_CMD = b"RESET"

    def __init__(self, net: NetworkConfig, model: ModelConfig, det: DetectionConfig):
        self.extractor = FaceEyeExtractor(crop_size=model.img_size)
        self.classifier = EyeStateClassifier(model)
        self.analyzer = DrowsinessAnalyzer(det)

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.bind(f"tcp://*:{net.port}")
        print(f"Listening on port {net.port}")

    def _process(self, frame: np.ndarray) -> AnalysisResult:
        obs = self.extractor.process(frame)
        p_closed = None
        if obs.face_found:
            p_closed = self.classifier.predict_closed_prob([obs.left_crop, obs.right_crop])
        return self.analyzer.update(obs, p_closed, time.monotonic())

    def run(self) -> None:
        try:
            while True:
                msg = self._sock.recv()
                if msg == self.RESET_CMD:
                    self.analyzer.reset()
                    self._sock.send_json({"ok": True})
                    print("Analyzer reset: recalibrating")
                    continue

                frame = cv2.imdecode(np.frombuffer(msg, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    self._sock.send_json({"error": "bad frame"})
                    continue
                t0 = time.monotonic()
                result = self._process(frame).to_dict()
                self._sock.send_json(result)
                print(f"frame processed in {time.monotonic() - t0:.2f}s")
        except KeyboardInterrupt:
            print("Shutting down")
        finally:
            self.extractor.close()
            self._sock.close(linger=0)
