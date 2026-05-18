"""
Lightweight mel spectrogram utilities for audio-visual alignment.

Complements sync_detector.py's Whisper-based mel with a librosa-based
alternative that has no model download requirement — useful for fast
testing, unit tests, and preprocessing pipelines.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


def extract_mel(
    audio_path: str,
    sr: int = 16000,
    n_mels: int = 80,
    hop_length: int = 160,
    win_length: int = 400,
    fmin: float = 0.0,
    fmax: Optional[float] = 8000.0,
) -> np.ndarray:
    """
    Compute log-mel spectrogram from a WAV file.

    Parameters
    ----------
    audio_path  : path to mono WAV file (16kHz recommended)
    sr          : target sample rate; librosa resamples if needed
    n_mels      : number of mel filter banks (80 matches Whisper / SyncNet)
    hop_length  : STFT hop in samples (160 @ 16kHz = 10ms)
    win_length  : STFT window in samples (400 @ 16kHz = 25ms)
    fmin/fmax   : mel filter bank frequency bounds

    Returns
    -------
    np.ndarray of shape (n_mels, T) in dB scale
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required: pip install librosa")

    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax,
    )
    return librosa.power_to_db(mel, ref=np.max)  # (n_mels, T)


def align_to_frames(
    mel: np.ndarray,
    n_video_frames: int,
    video_fps: float = 25.0,
    audio_sr: int = 16000,
    hop_length: int = 160,
) -> np.ndarray:
    """
    Resample mel time axis to align with video frame count.

    Each output column corresponds to one video frame, enabling
    per-frame audio-visual feature pairing without interpolation artefacts.

    Parameters
    ----------
    mel            : (n_mels, T_mel) log-mel spectrogram
    n_video_frames : number of video frames to align to
    video_fps      : video frame rate (default 25fps)
    audio_sr       : audio sample rate (default 16kHz)
    hop_length     : STFT hop used when computing mel

    Returns
    -------
    np.ndarray of shape (n_mels, n_video_frames)
    """
    T_mel = mel.shape[1]
    if n_video_frames <= 0 or T_mel == 0:
        return np.zeros((mel.shape[0], max(n_video_frames, 0)), dtype=mel.dtype)

    indices = np.linspace(0, T_mel - 1, n_video_frames).astype(int)
    return mel[:, indices]


def segment_mel(
    mel: np.ndarray,
    n_video_frames: int,
    window_frames: int,
    hop_frames: int,
) -> list[np.ndarray]:
    """
    Slice aligned mel into sliding windows matching video frame windows.

    Useful for per-window SyncNet inference without loading full mel each time.

    Parameters
    ----------
    mel            : (n_mels, n_video_frames) aligned mel (from align_to_frames)
    n_video_frames : total number of video frames (= mel.shape[1])
    window_frames  : number of frames per window
    hop_frames     : stride in frames between windows

    Returns
    -------
    List of (n_mels, window_frames) arrays, one per window
    """
    windows = []
    for start in range(0, n_video_frames - window_frames + 1, hop_frames):
        windows.append(mel[:, start : start + window_frames])
    return windows
