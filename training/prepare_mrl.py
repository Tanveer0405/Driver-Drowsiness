"""Sort the MRL Eye Dataset into open/ and closed/ folders.

MRL filenames look like: s0001_00123_0_0_1_0_0_01.png
The 5th field (index 4) is the eye state: 0 = closed, 1 = open.

Usage:
    python -m training.prepare_mrl --src /kaggle/input/mrl-eye-dataset --dst data/eyes
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


class MRLPreparer:
    def __init__(self, src: Path, dst: Path, limit_per_class: int | None = None):
        self.src, self.dst, self.limit = src, dst, limit_per_class

    def run(self) -> None:
        counts = {"open": 0, "closed": 0}
        for cls in counts:
            (self.dst / cls).mkdir(parents=True, exist_ok=True)

        for path in self.src.rglob("*.png"):
            parts = path.stem.split("_")
            if len(parts) < 5 or parts[4] not in ("0", "1"):
                continue
            cls = "open" if parts[4] == "1" else "closed"
            if self.limit and counts[cls] >= self.limit:
                continue
            shutil.copy(path, self.dst / cls / path.name)
            counts[cls] += 1
        print("Copied:", counts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, default=Path("data/eyes"))
    ap.add_argument("--limit", type=int, default=20000, help="max images per class")
    a = ap.parse_args()
    MRLPreparer(a.src, a.dst, a.limit).run()
