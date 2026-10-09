"""Keras -> TFLite conversion and a quick accuracy sanity check."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf


class TFLiteExporter:
    def __init__(self, quantization: str = "dynamic"):
        self.quantization = quantization

    def export(self, model: tf.keras.Model, out_path: Path) -> Path:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        if self.quantization in ("dynamic", "float16"):
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
        if self.quantization == "float16":
            conv.target_spec.supported_types = [tf.float16]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(conv.convert())
        print(f"Saved {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
        return out_path

    @staticmethod
    def verify(tflite_path: Path, val_ds: tf.data.Dataset, max_samples: int = 500) -> float:
        """Run the .tflite on validation images; returns accuracy."""
        interp = tf.lite.Interpreter(model_path=str(tflite_path))
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]["index"]
        out = interp.get_output_details()[0]["index"]

        correct = total = 0
        for images, labels in val_ds:
            for img, lab in zip(images.numpy(), labels.numpy()):
                interp.set_tensor(inp, img[None].astype(np.float32))
                interp.invoke()
                pred = float(interp.get_tensor(out)[0][0]) > 0.5
                correct += int(pred == bool(lab[0]))
                total += 1
                if total >= max_samples:
                    acc = correct / total
                    print(f"TFLite accuracy on {total} val samples: {acc:.3f}")
                    return acc
        acc = correct / max(total, 1)
        print(f"TFLite accuracy on {total} val samples: {acc:.3f}")
        return acc
