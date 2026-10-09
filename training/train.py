"""Training entrypoint.

    python -m training.train --data_dir data/eyes --out_dir models
"""
from __future__ import annotations

import argparse

from common.config import TrainConfig
from training.trainer import Trainer


def main() -> None:
    d = TrainConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=d.data_dir)
    ap.add_argument("--out_dir", default=d.out_dir)
    ap.add_argument("--img_size", type=int, default=d.img_size)
    ap.add_argument("--batch_size", type=int, default=d.batch_size)
    ap.add_argument("--epochs_head", type=int, default=d.epochs_head)
    ap.add_argument("--epochs_finetune", type=int, default=d.epochs_finetune)
    ap.add_argument("--quantization", default=d.quantization,
                    choices=["dynamic", "float16", "none"])
    a = ap.parse_args()

    cfg = TrainConfig(
        data_dir=a.data_dir, out_dir=a.out_dir, img_size=a.img_size,
        batch_size=a.batch_size, epochs_head=a.epochs_head,
        epochs_finetune=a.epochs_finetune, quantization=a.quantization,
    )
    Trainer(cfg).run()


if __name__ == "__main__":
    main()
