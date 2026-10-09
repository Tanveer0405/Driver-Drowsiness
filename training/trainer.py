"""Orchestrates: data -> model -> 2-stage training -> TFLite export."""
from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from common.config import TrainConfig
from training.dataset import EyeDataset
from training.exporter import TFLiteExporter
from training.model_builder import EyeStateModelBuilder


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.out_dir = Path(cfg.out_dir)
        self.dataset = EyeDataset(cfg)
        self.builder = EyeStateModelBuilder(cfg)
        self.exporter = TFLiteExporter(cfg.quantization)

    def _callbacks(self):
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1),
        ]

    def run(self) -> Path:
        tf.keras.utils.set_random_seed(self.cfg.seed)
        train_ds, val_ds = self.dataset.build()
        model = self.builder.build()

        print("\n== Stage 1: train classifier head ==")
        model.fit(train_ds, validation_data=val_ds,
                  epochs=self.cfg.epochs_head, callbacks=self._callbacks())

        print("\n== Stage 2: fine-tune top layers ==")
        self.builder.unfreeze_top()
        model.fit(train_ds, validation_data=val_ds,
                  epochs=self.cfg.epochs_finetune, callbacks=self._callbacks())

        print("\n== Final evaluation ==")
        model.evaluate(val_ds)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        model.save(self.out_dir / "eye_state.keras")
        tflite_path = self.exporter.export(model, self.out_dir / "eye_state.tflite")
        self.exporter.verify(tflite_path, val_ds)
        return tflite_path
