"""
Unit tests for preprocessing.audio_features.

Tests synthesise WAV files programmatically using the wave module — no real
dataset or ffmpeg required. librosa must be installed (pip install librosa).
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

requires_librosa = pytest.mark.skipif(
    not LIBROSA_AVAILABLE, reason="librosa not installed"
)

from preprocessing.audio_features import extract_mel, align_to_frames, segment_mel


# ------------------------------------------------------------------ #
#  Fixture: synthetic WAV                                              #
# ------------------------------------------------------------------ #

def _write_sine_wav(path: str, freq: float = 440.0, sr: int = 16000,
                    duration: float = 2.0, amplitude: int = 16000) -> None:
    """Write a pure sine wave to a mono 16-bit WAV file using stdlib only."""
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    samples = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


@pytest.fixture
def sine_wav(tmp_path) -> str:
    p = str(tmp_path / "sine_440.wav")
    _write_sine_wav(p, freq=440.0, sr=16000, duration=2.0)
    return p


@pytest.fixture
def short_wav(tmp_path) -> str:
    p = str(tmp_path / "short.wav")
    _write_sine_wav(p, freq=220.0, sr=16000, duration=0.5)
    return p


# ------------------------------------------------------------------ #
#  extract_mel tests                                                   #
# ------------------------------------------------------------------ #

@requires_librosa
def test_extract_mel_shape(sine_wav):
    """extract_mel returns (n_mels, T) with correct first dimension."""
    mel = extract_mel(sine_wav, sr=16000, n_mels=80, hop_length=160, win_length=400)
    assert mel.ndim == 2
    assert mel.shape[0] == 80, f"Expected 80 mel bins, got {mel.shape[0]}"
    # 2 seconds at 16kHz with hop=160 → T ≈ 200 frames (within ±5)
    assert 190 <= mel.shape[1] <= 210, f"Unexpected time frames: {mel.shape[1]}"


@requires_librosa
def test_extract_mel_dtype(sine_wav):
    """extract_mel returns float32/float64 (not integer) values."""
    mel = extract_mel(sine_wav)
    assert np.issubdtype(mel.dtype, np.floating)


@requires_librosa
def test_extract_mel_db_range(sine_wav):
    """extract_mel returns dB-scaled values (negative for power in dB, max ≤ 0)."""
    mel = extract_mel(sine_wav)
    # power_to_db with ref=max always produces max=0
    assert mel.max() <= 1.0   # may be exactly 0.0 for pure sine


@requires_librosa
def test_extract_mel_short_audio(short_wav):
    """extract_mel handles short audio (<1s) without error."""
    mel = extract_mel(short_wav, sr=16000, n_mels=40)
    assert mel.shape[0] == 40
    assert mel.shape[1] > 0


@requires_librosa
def test_extract_mel_custom_n_mels(sine_wav):
    """n_mels parameter is respected."""
    for n_mels in [40, 64, 128]:
        mel = extract_mel(sine_wav, n_mels=n_mels)
        assert mel.shape[0] == n_mels, f"n_mels={n_mels} not respected"


# ------------------------------------------------------------------ #
#  align_to_frames tests                                               #
# ------------------------------------------------------------------ #

@requires_librosa
def test_align_to_frames_shape(sine_wav):
    """align_to_frames returns (n_mels, n_video_frames)."""
    mel = extract_mel(sine_wav, n_mels=80)
    aligned = align_to_frames(mel, n_video_frames=50)
    assert aligned.shape == (80, 50), f"Expected (80, 50), got {aligned.shape}"


def test_align_to_frames_no_librosa():
    """align_to_frames works without librosa (pure numpy)."""
    mel = np.random.rand(80, 200).astype(np.float32)
    aligned = align_to_frames(mel, n_video_frames=50)
    assert aligned.shape == (80, 50)


def test_align_to_frames_exact_match():
    """align_to_frames with n_video_frames == T_mel is identity-like."""
    mel = np.arange(80 * 100, dtype=np.float32).reshape(80, 100)
    aligned = align_to_frames(mel, n_video_frames=100)
    assert aligned.shape == (80, 100)
    # First and last columns should match
    np.testing.assert_array_equal(aligned[:, 0], mel[:, 0])
    np.testing.assert_array_equal(aligned[:, -1], mel[:, -1])


def test_align_to_frames_zero_frames():
    """align_to_frames with 0 video frames returns (n_mels, 0) without error."""
    mel = np.zeros((80, 100))
    aligned = align_to_frames(mel, n_video_frames=0)
    assert aligned.shape == (80, 0)


# ------------------------------------------------------------------ #
#  segment_mel tests                                                   #
# ------------------------------------------------------------------ #

def test_segment_mel_count():
    """segment_mel returns the expected number of windows."""
    mel = np.zeros((80, 100))
    # 100 frames, window=25, hop=5 → (100-25)//5 + 1 = 16 windows
    windows = segment_mel(mel, n_video_frames=100, window_frames=25, hop_frames=5)
    expected = (100 - 25) // 5 + 1
    assert len(windows) == expected, f"Expected {expected} windows, got {len(windows)}"


def test_segment_mel_window_shape():
    """Each segment window has shape (n_mels, window_frames)."""
    mel = np.random.rand(80, 60).astype(np.float32)
    windows = segment_mel(mel, n_video_frames=60, window_frames=25, hop_frames=5)
    for w in windows:
        assert w.shape == (80, 25), f"Bad window shape: {w.shape}"


def test_segment_mel_empty_when_too_short():
    """segment_mel returns empty list when video is shorter than one window."""
    mel = np.zeros((80, 10))
    windows = segment_mel(mel, n_video_frames=10, window_frames=25, hop_frames=5)
    assert windows == []
