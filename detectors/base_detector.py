"""Abstract base class for all three detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from preprocessing.face_extractor import FaceTrack


class BaseDetector(ABC):
    """
    Common interface for visual / rPPG / sync detectors.

    Each detector receives a FaceTrack (already extracted by UnifiedFaceExtractor)
    and an optional audio path, then returns a fake probability in [0, 1]
    or None when the modality is unavailable (e.g., no audio for SyncDetector).
    """

    def __init__(self, config: dict):
        self.config = config
        self.device: str = config.get("device", "cuda")
        self._loaded: bool = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights. Called lazily on first detect()."""

    @abstractmethod
    def _detect_impl(
        self,
        face_track: FaceTrack,
        audio_path: Optional[str] = None,
    ) -> Optional[float]:
        """
        Core detection logic. Returns fake score in [0, 1] or None.

        Parameters
        ----------
        face_track  : pre-extracted FaceTrack from UnifiedFaceExtractor
        audio_path  : path to extracted WAV file (optional)
        """

    def detect(
        self,
        face_track: FaceTrack,
        audio_path: Optional[str] = None,
    ) -> Optional[float]:
        """Public API: lazy load + detect."""
        if not self._loaded:
            self.load()
            self._loaded = True
        return self._detect_impl(face_track, audio_path)

    async def detect_async(
        self,
        face_track: FaceTrack,
        audio_path: Optional[str] = None,
    ) -> Optional[float]:
        """Async wrapper for use with asyncio.gather()."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.detect, face_track, audio_path)
