"""Audio extraction utilities using FFmpeg."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional


class AudioExtractor:
    """Extract audio track from video using FFmpeg."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.sample_rate: int = config.get("sample_rate", 16000)
        self.channels: int = config.get("channels", 1)

    def has_audio(self, video_path: str) -> bool:
        """Check if video has an audio stream."""
        cmd = [
            "ffprobe", "-v", "quiet", "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return "audio" in result.stdout
        except Exception:
            return False

    def extract(self, video_path: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Extract audio to WAV file.

        Parameters
        ----------
        video_path : path to input video
        output_path : optional output WAV path. If None, writes to a temp file.

        Returns
        -------
        Path to WAV file, or None if no audio track found.
        """
        if not self.has_audio(video_path):
            return None

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = tmp.name
            tmp.close()

        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(self.sample_rate),
            "-ac", str(self.channels),
            str(output_path),
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=120)
            return output_path
        except subprocess.CalledProcessError:
            return None
        except subprocess.TimeoutExpired:
            return None

    def extract_to_temp(self, video_path: str) -> Optional[str]:
        """Extract audio to a temporary file. Caller must delete it."""
        return self.extract(video_path, output_path=None)
