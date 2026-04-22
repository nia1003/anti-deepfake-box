"""
Live input sources: webcam, screen capture, RTSP stream.
Yields BGR frames on demand for real-time inference.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Iterator, Optional

import cv2
import numpy as np


class FrameSource(ABC):
    @abstractmethod
    def frames(self) -> Iterator[np.ndarray]:
        """Yield BGR frames indefinitely."""

    def __enter__(self): return self
    def __exit__(self, *_): self.release()
    def release(self): pass


class WebcamSource(FrameSource):
    """USB/built-in webcam."""
    def __init__(self, device_id: int = 0, fps: float = 25.0):
        self.device_id = device_id
        self.fps = fps
        self._cap: Optional[cv2.VideoCapture] = None

    def frames(self) -> Iterator[np.ndarray]:
        self._cap = cv2.VideoCapture(self.device_id)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        interval = 1.0 / self.fps
        try:
            while True:
                t0 = time.time()
                ret, frame = self._cap.read()
                if not ret:
                    break
                yield frame
                elapsed = time.time() - t0
                if elapsed < interval:
                    time.sleep(interval - elapsed)
        finally:
            self.release()

    def release(self):
        if self._cap:
            self._cap.release()
            self._cap = None


class RTSPSource(FrameSource):
    """RTSP / IP camera stream with buffered read to prevent lag."""
    def __init__(self, url: str, fps: float = 25.0):
        self.url = url
        self.fps = fps
        self._latest: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False

    def _reader_thread(self, cap: cv2.VideoCapture):
        while self._running:
            ret, frame = cap.read()
            if ret:
                with self._lock:
                    self._latest = frame

    def frames(self) -> Iterator[np.ndarray]:
        cap = cv2.VideoCapture(self.url)
        self._running = True
        t = threading.Thread(target=self._reader_thread, args=(cap,), daemon=True)
        t.start()
        interval = 1.0 / self.fps
        try:
            while True:
                t0 = time.time()
                with self._lock:
                    frame = self._latest
                if frame is not None:
                    yield frame.copy()
                elapsed = time.time() - t0
                if elapsed < interval:
                    time.sleep(interval - elapsed)
        finally:
            self._running = False
            cap.release()

    def release(self):
        self._running = False


class ScreenSource(FrameSource):
    """Screen capture via mss (cross-platform)."""
    def __init__(self, fps: float = 5.0, monitor: int = 1):
        self.fps = fps
        self.monitor = monitor

    def frames(self) -> Iterator[np.ndarray]:
        try:
            import mss
            import mss.tools
        except ImportError:
            raise ImportError("pip install mss")

        interval = 1.0 / self.fps
        with mss.mss() as sct:
            mon = sct.monitors[self.monitor]
            while True:
                t0 = time.time()
                shot = sct.grab(mon)
                frame = np.array(shot)[:, :, :3]   # BGRA → BGR
                yield frame
                elapsed = time.time() - t0
                if elapsed < interval:
                    time.sleep(interval - elapsed)


def source_from_uri(uri: str, fps: float = 25.0) -> FrameSource:
    """
    Factory:
      '0', '1', ... → WebcamSource(device_id)
      'rtsp://...'  → RTSPSource
      'screen'      → ScreenSource
    """
    if uri.isdigit():
        return WebcamSource(device_id=int(uri), fps=fps)
    if uri.lower().startswith("rtsp://") or uri.lower().startswith("http://"):
        return RTSPSource(url=uri, fps=fps)
    if uri.lower() == "screen":
        return ScreenSource(fps=min(fps, 10.0))
    raise ValueError(f"Unknown source URI: {uri}")
