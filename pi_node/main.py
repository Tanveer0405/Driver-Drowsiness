"""Run on the Raspberry Pi:  python -m pi_node.main --model models/eye_state.tflite"""
from __future__ import annotations

import argparse

from common.config import DetectionConfig, ModelConfig, NetworkConfig
from pi_node.server import DrowsinessServer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=ModelConfig.model_path)
    ap.add_argument("--port", type=int, default=NetworkConfig.port)
    ap.add_argument("--img_size", type=int, default=ModelConfig.img_size)
    a = ap.parse_args()
    DrowsinessServer(
        NetworkConfig(port=a.port),
        ModelConfig(model_path=a.model, img_size=a.img_size),
        DetectionConfig(),
    ).run()


if __name__ == "__main__":
    main()
