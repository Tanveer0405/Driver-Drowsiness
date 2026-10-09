"""Dataset loading + augmentation for the eye-state classifier."""
from __future__ import annotations

import tensorflow as tf

from common.config import TrainConfig


class EyeDataset:
    """Expects data_dir/open/*.png and data_dir/closed/*.png.

    Label convention: closed = 1, open = 0 (model outputs P(closed)).
    Images are loaded grayscale (matches the Pi's grayscale eye crops),
    then replicated to 3 channels for MobileNetV2. Pixel range stays 0-255;
    scaling to [-1, 1] happens inside the model.
    """

    CLASS_NAMES = ["open", "closed"]

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self._augment = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.08),
                tf.keras.layers.RandomZoom(0.10),
                tf.keras.layers.RandomBrightness(0.30, value_range=(0, 255)),
                tf.keras.layers.RandomContrast(0.30),
            ]
        )

    def _load(self, subset: str) -> tf.data.Dataset:
        c = self.cfg
        return tf.keras.utils.image_dataset_from_directory(
            c.data_dir,
            labels="inferred",
            label_mode="binary",
            class_names=self.CLASS_NAMES,
            color_mode="grayscale",
            image_size=(c.img_size, c.img_size),
            batch_size=c.batch_size,
            validation_split=c.val_split,
            subset=subset,
            seed=c.seed,
        )

    @staticmethod
    def _to_rgb(x, y):
        return tf.image.grayscale_to_rgb(x), y

    def _augment_fn(self, x, y):
        x = self._augment(x, training=True)
        return tf.clip_by_value(x, 0.0, 255.0), y

    def build(self):
        """Returns (train_ds, val_ds)."""
        auto = tf.data.AUTOTUNE
        train = self._load("training").map(self._to_rgb, num_parallel_calls=auto)
        train = train.map(self._augment_fn, num_parallel_calls=auto).prefetch(auto)
        val = self._load("validation").map(self._to_rgb, num_parallel_calls=auto)
        return train, val.prefetch(auto)
