"""
rPPG-based Deepfake Detector.

Principle: real human faces exhibit periodic blood volume pulse (PPG) signals
in subtle skin colour variations. Synthetic deepfake faces typically lack these
genuine physiological signals, resulting in a low signal-to-noise ratio (SNR)
when a pre-trained rPPG model tries to extract a waveform from them.

Pipeline:
1. FaceTrack.crops_128 (T, 128, 128, 3) → PhysNet → PPG waveform (T,)
2. Compute SNR via Welch PSD in the physiological HR band (0.75–3.5 Hz)
3. Map SNR → fake score via sigmoid: low SNR = more likely fake
4. Fallback to CHROM (unsupervised) when T < 30

SNR threshold is calibrated on FF++ validation set via
scripts/calibrate_snr.py (Youden's J statistic).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .base_detector import BaseDetector
from preprocessing.face_extractor import FaceTrack

_RPPG_PATH = Path(__file__).parent.parent / "third_party" / "rppg-toolbox"
if _RPPG_PATH.exists():
    sys.path.insert(0, str(_RPPG_PATH))


# ------------------------------------------------------------------ #
#  SNR calculation                                                     #
# ------------------------------------------------------------------ #

def compute_ppg_snr(
    ppg: np.ndarray,
    fps: float = 25.0,
    hr_low: float = 0.75,
    hr_high: float = 3.5,
) -> float:
    """
    Compute signal-to-noise ratio of a PPG waveform.

    Parameters
    ----------
    ppg      : 1-D waveform, length T
    fps      : frames per second
    hr_low   : lower bound of physiological HR band (Hz), corresponds to ~45 bpm
    hr_high  : upper bound of physiological HR band (Hz), corresponds to ~210 bpm

    Returns
    -------
    SNR in dB (float). Higher = more physiologically plausible = more likely real.
    """
    from scipy import signal as sp_signal

    T = len(ppg)
    if T < 8:
        return -np.inf

    # Welch PSD
    nperseg = min(T, 256)
    freqs, psd = sp_signal.welch(ppg, fs=fps, nperseg=nperseg)

    # Signal power in HR band
    hr_mask = (freqs >= hr_low) & (freqs <= hr_high)
    noise_mask = ~hr_mask & (freqs > 0)

    signal_power = float(psd[hr_mask].max()) if hr_mask.any() else 0.0
    noise_power = float(psd[noise_mask].mean()) if noise_mask.any() else 1e-6
    noise_power = max(noise_power, 1e-9)

    snr_db = 10.0 * np.log10(signal_power / noise_power + 1e-9)
    return snr_db


def snr_to_fake_score(snr: float, threshold: float = 1.5, scale: float = 1.0) -> float:
    """
    Map SNR to fake probability via reverse sigmoid.

    snr < threshold  → fake_score → 1   (low SNR = likely fake)
    snr >> threshold → fake_score → 0   (high SNR = likely real)
    """
    x = -(snr - threshold) * scale
    return float(1.0 / (1.0 + np.exp(-x)))


# ------------------------------------------------------------------ #
#  CHROM fallback (unsupervised, no model required)                    #
# ------------------------------------------------------------------ #

def chrom_ppg(crops_rgb: np.ndarray) -> np.ndarray:
    """
    CHROM unsupervised rPPG algorithm.
    Reference: De Haan & Jeanne (2013).

    Parameters
    ----------
    crops_rgb : (T, H, W, 3) uint8

    Returns
    -------
    ppg : (T,) float32
    """
    T = len(crops_rgb)
    rgb = crops_rgb.astype(np.float32).reshape(T, -1, 3)
    mean_rgb = rgb.mean(axis=1)  # (T, 3)

    # Normalise by temporal mean
    mean_rgb_bar = mean_rgb.mean(axis=0, keepdims=True) + 1e-6
    cn = mean_rgb / mean_rgb_bar  # (T, 3)

    Xs = 3 * cn[:, 0] - 2 * cn[:, 1]
    Ys = 1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2]

    std_xs = np.std(Xs) + 1e-6
    std_ys = np.std(Ys) + 1e-6
    alpha = std_xs / std_ys

    ppg = Xs - alpha * Ys
    ppg -= ppg.mean()
    return ppg.astype(np.float32)


# ------------------------------------------------------------------ #
#  Detector class                                                      #
# ------------------------------------------------------------------ #

class RPPGDetector(BaseDetector):
    """
    PhysNet-based rPPG detector for deepfake identification.

    Uses SNR of extracted PPG waveform as a proxy for physiological plausibility.
    Low SNR on deepfake faces → high fake score.
    """

    MIN_FRAMES_FOR_NEURAL = 30
    CHUNK_SIZE = 180

    def __init__(self, config: dict):
        super().__init__(config)
        self.pretrained_path: str = config.get("rppg_pretrained", "")
        self.snr_threshold: float = config.get("snr_threshold", 1.5)
        self.snr_scale: float = config.get("snr_scale", 1.0)
        self.model = None

    def load(self) -> None:
        try:
            from neural_methods.model.PhysNet import PhysNet
            self.model = PhysNet(S=2, in_ch=3, out_ch=1)
            if self.pretrained_path and Path(self.pretrained_path).exists():
                state = torch.load(self.pretrained_path, map_location="cpu")
                if isinstance(state, dict) and "model_state_dict" in state:
                    state = state["model_state_dict"]
                self.model.load_state_dict(state, strict=False)
            self.model.eval()
            self.model = self.model.to(self.device)
        except ImportError:
            self.model = None  # Will use CHROM fallback

    def _extract_ppg_neural(self, crops_128: np.ndarray) -> np.ndarray:
        """PhysNet forward pass: (T, 128, 128, 3) → (T,) PPG waveform."""
        # PhysNet expects (B, C, T, H, W) with T = CHUNK_SIZE
        T = len(crops_128)
        chunk = crops_128[:self.CHUNK_SIZE] if T >= self.CHUNK_SIZE else crops_128

        # Pad if needed
        if len(chunk) < self.CHUNK_SIZE:
            pad = np.zeros((self.CHUNK_SIZE - len(chunk), 128, 128, 3), np.uint8)
            chunk = np.concatenate([chunk, pad], axis=0)

        # (T, 128, 128, 3) → (1, 3, T, 128, 128) float32 [0,1]
        x = chunk.astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(3, 0, 1, 2).unsqueeze(0)  # (1, 3, T, H, W)
        x = x.to(self.device)

        with torch.no_grad():
            rppg = self.model(x)  # (1, T) or (1, 1, T)
            rppg = rppg.squeeze().cpu().numpy()

        return rppg[:T].astype(np.float32)

    def _detect_impl(
        self,
        face_track: FaceTrack,
        audio_path: Optional[str] = None,
    ) -> Optional[float]:
        if face_track is None or face_track.T == 0:
            return None

        crops_128 = face_track.crops_128
        T = face_track.T
        fps = face_track.fps

        if self.model is not None and T >= self.MIN_FRAMES_FOR_NEURAL:
            ppg = self._extract_ppg_neural(crops_128)
        else:
            ppg = chrom_ppg(crops_128)

        snr = compute_ppg_snr(ppg, fps=fps)
        return snr_to_fake_score(snr, self.snr_threshold, self.snr_scale)

    def get_ppg_and_snr(self, face_track: FaceTrack) -> dict:
        """
        Debug/calibration helper: return raw PPG waveform + SNR + fake score.
        """
        if not self._loaded:
            self.load()
            self._loaded = True

        crops_128 = face_track.crops_128
        T = face_track.T
        fps = face_track.fps

        if self.model is not None and T >= self.MIN_FRAMES_FOR_NEURAL:
            ppg = self._extract_ppg_neural(crops_128)
            method = "PhysNet"
        else:
            ppg = chrom_ppg(crops_128)
            method = "CHROM"

        snr = compute_ppg_snr(ppg, fps=fps)
        fake_score = snr_to_fake_score(snr, self.snr_threshold, self.snr_scale)

        return {
            "ppg": ppg,
            "snr_db": snr,
            "fake_score": fake_score,
            "method": method,
            "num_frames": T,
        }
