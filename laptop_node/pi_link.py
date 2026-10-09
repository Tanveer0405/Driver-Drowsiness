"""ZeroMQ client that sends frames to the Pi and returns its JSON reply."""
from __future__ import annotations

import json
from typing import Optional

import cv2
import zmq

from common.config import NetworkConfig


class PiLink:
    def __init__(self, net: NetworkConfig):
        self._net = net
        self._ctx = zmq.Context.instance()
        self._sock = None
        self._connect()

    def _connect(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.connect(f"tcp://{self._net.pi_host}:{self._net.port}")

    def _request(self, payload: bytes) -> Optional[dict]:
        self._sock.send(payload)
        if self._sock.poll(self._net.reply_timeout_ms):
            return json.loads(self._sock.recv())
        self._connect()          # timed out: rebuild socket
        return None

    def analyze(self, frame) -> Optional[dict]:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._net.jpeg_quality])
        return self._request(buf.tobytes()) if ok else None

    def analyze_jpeg(self, jpeg: bytes) -> Optional[dict]:
        """Forward an already-encoded JPEG (used by the web dashboard)."""
        return self._request(jpeg)

    def reset(self) -> Optional[dict]:
        return self._request(b"RESET")
