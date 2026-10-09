"""Web dashboard. The laptop webcam (OpenCV) is streamed to the browser and relayed to the Pi.

    python -m web_node.app --host sct2026.local      then open http://localhost:8000
"""
from __future__ import annotations

import argparse
import threading
import time
import webbrowser
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, request, send_from_directory

from common.config import NetworkConfig
from laptop_node.pi_link import PiLink


class Camera:
    """Grabs frames in a background thread and keeps the latest one as JPEG."""

    def __init__(self, index: int = 0, w: int = 640, h: int = 480, quality: int = 70):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise SystemExit(f"Cannot open camera {index}. Allow camera access for your terminal "
                             "in System Settings > Privacy & Security > Camera.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.q = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        self.jpeg: bytes | None = None
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            ok, buf = cv2.imencode(".jpg", frame, self.q)
            if ok:
                with self.lock:
                    self.jpeg = buf.tobytes()
            time.sleep(0.01)

    def get(self) -> bytes | None:
        with self.lock:
            return self.jpeg


def create_app(link: PiLink, cam: Camera) -> Flask:
    app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"))
    lock = threading.Lock()                    # one request to the Pi at a time

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/video")
    def video():
        def gen():
            while True:
                j = cam.get()
                if j:
                    yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n"
                time.sleep(0.04)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")

    @app.post("/api/analyze")
    def analyze():
        jpeg = cam.get()
        if not jpeg:
            return jsonify({"error": "no_camera"})
        with lock:
            res = link._request(jpeg)
        return jsonify(res if res and "level" in res else {"error": "pi_offline"})

    @app.post("/api/reset")
    def reset():
        with lock:
            res = link.reset()
        return jsonify({"ok": bool(res)})

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=NetworkConfig.pi_host, help="Pi IP address or hostname")
    ap.add_argument("--web_port", type=int, default=8000)
    ap.add_argument("--camera", type=int, default=0)
    a = ap.parse_args()
    cam = Camera(a.camera)
    app = create_app(PiLink(NetworkConfig(pi_host=a.host)), cam)
    threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{a.web_port}")).start()
    app.run(host="127.0.0.1", port=a.web_port, threaded=True)


if __name__ == "__main__":
    main()