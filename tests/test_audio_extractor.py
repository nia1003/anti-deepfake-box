"""
Unit tests for preprocessing.audio_extractor.AudioExtractor.

Tests synthesise minimal video/audio files with ffmpeg so no real dataset
is needed. All tests are self-contained and can run with only ffmpeg + python.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import wave
from pathlib import Path

import pytest

# Ensure project root is in path when run from repo root or tests/
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.audio_extractor import AudioExtractor


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=5)
        return True
    except Exception:
        return False


def _ffprobe_available() -> bool:
    try:
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True, timeout=5)
        return True
    except Exception:
        return False


requires_ffmpeg = pytest.mark.skipif(
    not (_ffmpeg_available() and _ffprobe_available()),
    reason="ffmpeg/ffprobe not installed",
)


def _make_video_with_audio(path: str, duration: float = 2.0) -> None:
    """Create a tiny MP4 with a 440Hz sine audio track using ffmpeg lavfi."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        # sine audio source
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
        # black video source
        "-f", "lavfi", "-i", f"color=c=black:size=64x64:duration={duration}:rate=10",
        "-c:a", "aac", "-b:a", "32k",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "51",
        "-shortest",
        str(path),
    ]
    subprocess.run(cmd, check=True, timeout=30)


def _make_video_without_audio(path: str, duration: float = 2.0) -> None:
    """Create a tiny silent MP4 (no audio stream) using ffmpeg lavfi."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c=black:size=64x64:duration={duration}:rate=10",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "51",
        str(path),
    ]
    subprocess.run(cmd, check=True, timeout=30)


# ------------------------------------------------------------------ #
#  Tests                                                               #
# ------------------------------------------------------------------ #

@requires_ffmpeg
def test_has_audio_true(tmp_path):
    """has_audio() returns True for a video that contains an audio stream."""
    vp = tmp_path / "with_audio.mp4"
    _make_video_with_audio(str(vp))
    ext = AudioExtractor()
    assert ext.has_audio(str(vp)) is True


@requires_ffmpeg
def test_has_audio_false(tmp_path):
    """has_audio() returns False for a video with no audio track."""
    vp = tmp_path / "silent.mp4"
    _make_video_without_audio(str(vp))
    ext = AudioExtractor()
    assert ext.has_audio(str(vp)) is False


@requires_ffmpeg
def test_has_audio_nonexistent():
    """has_audio() returns False for a path that doesn't exist."""
    ext = AudioExtractor()
    assert ext.has_audio("/nonexistent/video.mp4") is False


@requires_ffmpeg
def test_extract_returns_wav(tmp_path):
    """extract() returns a valid WAV file path for a video with audio."""
    vp = tmp_path / "with_audio.mp4"
    _make_video_with_audio(str(vp))
    out_wav = str(tmp_path / "out.wav")

    ext = AudioExtractor({"sample_rate": 16000, "channels": 1})
    result = ext.extract(str(vp), output_path=out_wav)

    assert result is not None, "extract() returned None for a video with audio"
    assert Path(result).exists(), f"WAV file not found at {result}"
    assert Path(result).suffix == ".wav"

    # Validate the WAV header
    with wave.open(result) as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000
        assert wf.getnframes() > 0


@requires_ffmpeg
def test_extract_returns_none_for_silent(tmp_path):
    """extract() returns None when video has no audio."""
    vp = tmp_path / "silent.mp4"
    _make_video_without_audio(str(vp))

    ext = AudioExtractor()
    assert ext.extract(str(vp)) is None


@requires_ffmpeg
def test_extract_to_temp_creates_file(tmp_path):
    """extract_to_temp() creates a temp WAV file and returns its path."""
    vp = tmp_path / "with_audio.mp4"
    _make_video_with_audio(str(vp))

    ext = AudioExtractor()
    wav_path = ext.extract_to_temp(str(vp))

    assert wav_path is not None
    assert Path(wav_path).exists()

    # Caller is responsible for cleanup
    os.unlink(wav_path)
    assert not Path(wav_path).exists(), "Temp file should be deleted by caller"


@requires_ffmpeg
def test_extract_to_temp_returns_none_for_silent(tmp_path):
    """extract_to_temp() returns None for videos without audio."""
    vp = tmp_path / "silent.mp4"
    _make_video_without_audio(str(vp))

    ext = AudioExtractor()
    assert ext.extract_to_temp(str(vp)) is None


@requires_ffmpeg
def test_custom_sample_rate(tmp_path):
    """AudioExtractor respects the sample_rate config option."""
    vp = tmp_path / "with_audio.mp4"
    _make_video_with_audio(str(vp))
    out_wav = str(tmp_path / "out_22k.wav")

    ext = AudioExtractor({"sample_rate": 22050, "channels": 1})
    result = ext.extract(str(vp), output_path=out_wav)

    assert result is not None
    with wave.open(result) as wf:
        assert wf.getframerate() == 22050


def test_instantiation_defaults():
    """AudioExtractor can be instantiated with no config."""
    ext = AudioExtractor()
    assert ext.sample_rate == 16000
    assert ext.channels == 1
