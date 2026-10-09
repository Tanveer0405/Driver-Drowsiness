"""TFLite wrapper: eye crop(s) -> P(closed)."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from common.config import ModelConfig


def _load_interpreter_class():
    """Prefer the light runtime on the Pi; fall back to full TensorFlow."""
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    import tensorflow as tf
    return tf.lite.Interpreter


class EyeStateClassifier:
    def __init__(self, cfg: ModelConfig):
        Interpreter = _load_interpreter_class()
        self._interp = Interpreter(model_path=cfg.model_path, num_threads=cfg.num_threads)
        self._interp.allocate_tensors()
        self._in = self._interp.get_input_details()[0]
        self._out = self._interp.get_output_details()[0]
        self._size = cfg.img_size

    def _prepare(self, gray: np.ndarray) -> np.ndarray:
        """Grayscale uint8 (S,S) -> float32 (1,S,S,3) in 0-255 (model rescales itself)."""
        rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
        return rgb[None]

    def _predict_one(self, gray: np.ndarray) -> float:
        self._interp.set_tensor(self._in["index"], self._prepare(gray))
        self._interp.invoke()
        return float(self._interp.get_tensor(self._out["index"])[0][0])

    def predict_closed_prob(self, crops: List[np.ndarray]) -> Optional[float]:
        """Average P(closed) over the available eye crops."""
        crops = [c for c in crops if c is not None]
        if not crops:
            return None
        return float(np.mean([self._predict_one(c) for c in crops]))
