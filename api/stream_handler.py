"""
StreamHandler — per-session frame buffer with temporal smoothing.

Maintains a sliding window of detection scores and applies exponential
moving average (EMA) to reduce jitter in real-time display.
Also tracks frame quality via Laplacian variance (from PREVERVECT).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class FrameResult:
    timestamp: float
    fake_score: Optional[float]        # None = no face detected
    scores: dict                        # per-modality scores
    quality: float                      # Laplacian variance (focus metric)
    face_detected: bool
    latency_ms: float


@dataclass
class StreamState:
    """Per-session state for temporal smoothing and FPS tracking."""
    session_id: str
    smoothed_score: float = 0.5
    ema_alpha: float = 0.25            # lower = smoother, more lag
    history: deque = field(default_factory=lambda: deque(maxlen=30))
    frame_times: deque = field(default_factory=lambda: deque(maxlen=90))
    last_result: Optional[FrameResult] = None

    @property
    def fps(self) -> float:
        if len(self.frame_times) < 2:
            return 0.0
        diffs = [self.frame_times[i+1] - self.frame_times[i]
                 for i in range(len(self.frame_times) - 1)]
        return 1.0 / (np.mean(diffs) + 1e-6)

    @property
    def avg_score(self) -> float:
        if not self.history:
            return 0.5
        scores = [r.fake_score for r in self.history if r.fake_score is not None]
        return float(np.mean(scores)) if scores else 0.5

    def update(self, result: FrameResult) -> None:
        now = time.time()
        self.frame_times.append(now)
        self.history.append(result)
        self.last_result = result

        if result.fake_score is not None:
            # EMA smoothing — reduce weight when quality is low
            quality_w = min(result.quality / 200.0, 1.0)   # normalize Laplacian
            effective_alpha = self.ema_alpha * quality_w
            self.smoothed_score = (
                effective_alpha * result.fake_score
                + (1 - effective_alpha) * self.smoothed_score
            )

    def to_response(self) -> dict:
        r = self.last_result
        return {
            "smoothed_score": round(self.smoothed_score, 4),
            "avg_score":      round(self.avg_score, 4),
            "is_fake":        self.smoothed_score >= 0.5,
            "confidence":     self._confidence_label(),
            "fps":            round(self.fps, 1),
            "face_detected":  r.face_detected if r else False,
            "quality":        round(r.quality, 1) if r else 0.0,
            "latency_ms":     round(r.latency_ms, 1) if r else 0.0,
            "modality_scores": r.scores if r else {},
        }

    def _confidence_label(self) -> str:
        s = self.smoothed_score
        if s > 0.80 or s < 0.20:
            return "high"
        if s > 0.65 or s < 0.35:
            return "medium"
        return "low"


def laplacian_quality(frame_bgr: np.ndarray) -> float:
    """Sharpness metric via Laplacian variance. Low (<120) = blurry."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class SessionManager:
    """Thread-safe store of per-session StreamState objects."""

    def __init__(self) -> None:
        self._sessions: dict[str, StreamState] = {}

    def get_or_create(self, session_id: str) -> StreamState:
        if session_id not in self._sessions:
            self._sessions[session_id] = StreamState(session_id=session_id)
        return self._sessions[session_id]

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def active_count(self) -> int:
        return len(self._sessions)

    def all_states(self) -> dict:
        """Return a snapshot of all active session states for dashboard polling."""
        return {
            sid: state.to_response()
            for sid, state in self._sessions.items()
        }
