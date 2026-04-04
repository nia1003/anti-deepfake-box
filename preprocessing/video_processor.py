"""Video processing utilities: frame reading, chunking, metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np


def get_video_info(video_path: str) -> dict:
    """Return basic video metadata: fps, total_frames, width, height."""
    cap = cv2.VideoCapture(str(video_path))
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def read_frames_at_fps(video_path: str, fps_target: float = 25.0) -> Tuple[np.ndarray, float, List[int]]:
    """
    Read video frames sampled at fps_target.

    Returns (frames_bgr, native_fps, frame_indices).
    frames_bgr: (N, H, W, 3) uint8
    """
    cap = cv2.VideoCapture(str(video_path))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(native_fps / fps_target))
    target_indices = set(range(0, total, step))
    frames, indices = [], []
    fi = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if fi in target_indices:
            frames.append(frame)
            indices.append(fi)
        fi += 1
    cap.release()
    if not frames:
        return np.empty((0, 0, 0, 3), np.uint8), native_fps, []
    return np.stack(frames), native_fps, indices


def chunk_frames(frames: np.ndarray, chunk_size: int = 180, overlap: int = 0) -> List[np.ndarray]:
    """Split frame array into chunks of chunk_size with optional overlap."""
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(frames), step):
        chunk = frames[start: start + chunk_size]
        if len(chunk) > 0:
            chunks.append(chunk)
    return chunks


class VideoProcessor:
    """High-level video processing interface."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.fps_target = config.get("fps_target", 25.0)
        self.chunk_size = config.get("chunk_size", 180)
        self.chunk_overlap = config.get("chunk_overlap", 0)

    def read(self, video_path: str) -> Tuple[np.ndarray, float, List[int]]:
        return read_frames_at_fps(video_path, self.fps_target)

    def read_chunks(self, video_path: str) -> List[np.ndarray]:
        frames, _, _ = self.read(video_path)
        return chunk_frames(frames, self.chunk_size, self.chunk_overlap)

    def info(self, video_path: str) -> dict:
        return get_video_info(video_path)
