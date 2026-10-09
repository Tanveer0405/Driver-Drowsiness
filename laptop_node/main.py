"""Run on the laptop:  python -m laptop_node.main --host <pi-ip>
Keys: q = quit, r = recalibrate."""
from __future__ import annotations

import argparse
import time

import cv2

from common.config import NetworkConfig
from common.types import AlertLevel
from laptop_node.alarm import AlarmPlayer
from laptop_node.overlay import StatusOverlay
from laptop_node.pi_link import PiLink
from laptop_node.webcam import Webcam


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=NetworkConfig.pi_host)
    ap.add_argument("--camera", type=int, default=0)
    a = ap.parse_args()

    net = NetworkConfig(pi_host=a.host)
    cam, link = Webcam(a.camera, net.frame_width, net.frame_height), PiLink(net)
    alarm, overlay = AlarmPlayer(), StatusOverlay()
    period = 1.0 / net.target_fps

    try:
        while True:
            t0 = time.monotonic()
            ok, frame = cam.read()
            if not ok:
                break

            result = link.analyze(frame)
            link_ok = result is not None and "level" in result
            # Fail-safe: if the Pi stops answering, warn the driver.
            level = AlertLevel(result["level"]) if link_ok else AlertLevel.WARNING
            alarm.update(level)

            cv2.imshow("Driver monitor", overlay.draw(cv2.flip(frame, 1), result, level, link_ok))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                link.reset()

            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
