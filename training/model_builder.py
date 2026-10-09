"""Builds the MobileNetV2 eye-state model (output = P(closed))."""
from __future__ import annotations

import tensorflow as tf

from common.config import TrainConfig


class EyeStateModelBuilder:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.base: tf.keras.Model | None = None
        self.model: tf.keras.Model | None = None

    def build(self) -> tf.keras.Model:
        c = self.cfg
        shape = (c.img_size, c.img_size, 3)

        inputs = tf.keras.Input(shape=shape, name="eye_crop")      # float 0-255
        x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(inputs)

        self.base = tf.keras.applications.MobileNetV2(
            input_shape=shape, alpha=c.alpha, include_top=False, weights="imagenet"
        )
        self.base.trainable = False
        x = self.base(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="p_closed")(x)

        self.model = tf.keras.Model(inputs, outputs, name="eye_state")
        self.compile(c.lr_head)
        return self.model

    def unfreeze_top(self) -> None:
        """Fine-tuning stage: unfreeze the last N layers, keep BatchNorm frozen."""
        self.base.trainable = True
        for layer in self.base.layers[: -self.cfg.unfreeze_layers]:
            layer.trainable = False
        for layer in self.base.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        self.compile(self.cfg.lr_finetune)

    def compile(self, lr: float) -> None:
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="acc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )
