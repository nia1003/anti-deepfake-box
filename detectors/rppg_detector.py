"""
rPPG-based Deepfake Detector using POS (Plane-Orthogonal-to-Skin).

POS is an unsupervised signal processing algorithm — no checkpoint or GPU
required. It projects skin-colour vectors onto a plane orthogonal to the
illumination direction to isolate the blood volume pulse (BVP). Deepfake
faces lack genuine physiological periodicity, resulting in low SNR.

Reference: Wang et al. (2017). Algorithmic principles of remote PPG.
           IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .base_detector import BaseDetector
from preprocessing.face_extractor import FaceTrack


# ------------------------------------------------------------------ #
#  POS algorithm (Wang 2017)                                           #
# ------------------------------------------------------------------ #

def _pos_wang(frames: np.ndarray, fps: float) -> np.ndarray:
    """
    Parameters
    ----------
    frames : (T, H, W, 3) uint8 RGB
    fps    : frames per second

    Returns
    -------
    bvp : (T,) float64  bandpass-filtered blood volume pulse
    """
    from scipy import signal as sp_signal

    T = len(frames)
    l = max(2, math.ceil(1.6 * fps))  # sliding window length

    # Mean RGB per frame: (T, 3)
    rgb = frames.astype(np.float64).reshape(T, -1, 3).mean(axis=1)

    H = np.zeros(T, dtype=np.float64)
    _P = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])  # projection matrix

    for n in range(l, T):
        m = n - l
        chunk = rgb[m:n]                              # (l, 3)
        mean_c = chunk.mean(axis=0) + 1e-6
        Cn = (chunk / mean_c).T                       # (3, l)

        S = _P @ Cn                                   # (2, l)
        alpha = np.std(S[0]) / (np.std(S[1]) + 1e-9)
        h = S[0] + alpha * S[1]
        h -= h.mean()
        H[m:n] += h

    # Bandpass 0.75–3 Hz (physiological HR range: ~45–180 bpm)
    nyq = fps / 2.0
    b, a = sp_signal.butter(1, [0.75 / nyq, min(3.0 / nyq, 0.99)], btype='bandpass')
    bvp = sp_signal.filtfilt(b, a, H)
    return bvp


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
    Compute signal-to-noise ratio of a PPG/BVP waveform.

    Returns SNR in dB. Higher = more physiologically plausible = more likely real.
    """
    from scipy import signal as sp_signal

    T = len(ppg)
    if T < 8:
        return -np.inf

    nperseg = min(T, 256)
    freqs, psd = sp_signal.welch(ppg, fs=fps, nperseg=nperseg)

    hr_mask = (freqs >= hr_low) & (freqs <= hr_high)
    noise_mask = ~hr_mask & (freqs > 0)

    signal_power = float(psd[hr_mask].max()) if hr_mask.any() else 0.0
    noise_power = float(psd[noise_mask].mean()) if noise_mask.any() else 1e-6
    noise_power = max(noise_power, 1e-9)

    return 10.0 * np.log10(signal_power / noise_power + 1e-9)


def snr_to_fake_score(snr: float, threshold: float = 1.5, scale: float = 1.0) -> float:
    """
    Map SNR to fake probability via reverse sigmoid.

    snr << threshold → fake_score → 1  (low SNR = likely fake)
    snr >> threshold → fake_score → 0  (high SNR = likely real)
    """
    x = -(snr - threshold) * scale
    return float(1.0 / (1.0 + np.exp(-x)))


# ------------------------------------------------------------------ #
#  Detector class                                                      #
# ------------------------------------------------------------------ #

class RPPGDetector(BaseDetector):
    """
    POS-based rPPG detector for deepfake identification.

    No pretrained weights or GPU needed. Works on any number of frames.
    SNR threshold is calibrated on FF++ val set via scripts/calibrate_snr.py.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.snr_threshold: float = config.get("snr_threshold", 1.5)
        self.snr_scale: float = config.get("snr_scale", 1.0)

    def load(self) -> None:
        pass  # POS requires no model weights

    def _detect_impl(
        self,
        face_track: FaceTrack,
        audio_path: Optional[str] = None,
    ) -> Optional[float]:
        if face_track is None or face_track.T == 0:
            return None

        bvp = _pos_wang(face_track.crops_128, fps=face_track.fps)
        snr = compute_ppg_snr(bvp, fps=face_track.fps)
        return snr_to_fake_score(snr, self.snr_threshold, self.snr_scale)

    def get_ppg_and_snr(self, face_track: FaceTrack) -> dict:
        """Debug/calibration helper: return raw BVP waveform + SNR + fake score."""
        if not self._loaded:
            self.load()
            self._loaded = True

        bvp = _pos_wang(face_track.crops_128, fps=face_track.fps)
        snr = compute_ppg_snr(bvp, fps=face_track.fps)

        return {
            "ppg": bvp,
            "snr_db": snr,
            "fake_score": snr_to_fake_score(snr, self.snr_threshold, self.snr_scale),
            "method": "POS",
            "num_frames": face_track.T,
        }
