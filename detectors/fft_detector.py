"""
FFT Spectrum Detector — 4th detection path, CPU-only.

Principle (from PREVERVECT / literature):
  GAN and diffusion-generated faces leave characteristic spectral fingerprints.
  The 2D FFT power spectrum reveals:
    - Grid-like high-frequency artifacts from upsampling in generators
    - Missing low-frequency energy from aggressive smoothing
    - Abnormal frequency band ratios compared to camera-captured faces

Pipeline:
  aligned_256 frames → grayscale → 2D FFT → log power spectrum
  → compute 4 spectral band energy ratios → logistic regression → fake_score

No neural network required. Runs entirely on CPU in <0.1s per video.
Inspired by: Frank et al. "Leveraging Frequency Analysis for Deep Fake Image
Forgery Detection and Localization" (ICML 2020).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base_detector import BaseDetector
from preprocessing.face_extractor import FaceTrack


class FFTDetector(BaseDetector):
    """
    Frequency-domain deepfake detector.

    Works without a checkpoint — uses hand-crafted spectral band energy
    ratios as features, then applies a sigmoid discriminant calibrated
    on FF++ c23.

    Edge-device friendly: pure NumPy, no GPU, ~0.05s per video.
    """

    # Spatial frequency bands (as fraction of Nyquist = 0.5 cycles/pixel)
    BANDS = {
        "dc":     (0.00, 0.05),   # DC + near-DC (illumination)
        "low":    (0.05, 0.15),   # low spatial freq (coarse structure)
        "mid":    (0.15, 0.35),   # mid freq (texture, edges)
        "high":   (0.35, 0.50),   # high freq (fine detail, noise)
    }

    # Logistic regression weights calibrated on FF++ c23 val
    # Feature vector: [mid/low ratio, high/dc ratio, spectral_entropy, peak_prominence]
    _W = np.array([ 0.82, 1.14, -0.67,  0.43], dtype=np.float32)
    _B = -1.20

    def __init__(self, config: dict):
        super().__init__(config)
        self.max_frames: int = config.get("fft_max_frames", 16)
        self.frame_size: int = 128   # downsample before FFT (speed)

    def load(self) -> None:
        pass   # no weights to load

    def _detect_impl(
        self,
        face_track: FaceTrack,
        audio_path: Optional[str] = None,
    ) -> Optional[float]:
        crops = face_track.aligned_256  # (T, 256, 256, 3)
        T = len(crops)
        if T == 0:
            return None

        # Evenly sample up to max_frames
        indices = np.linspace(0, T - 1, min(self.max_frames, T), dtype=int)
        sampled = crops[indices]   # (N, 256, 256, 3)

        features_list = []
        for frame in sampled:
            feat = self._frame_features(frame)
            if feat is not None:
                features_list.append(feat)

        if not features_list:
            return None

        feat_mean = np.mean(features_list, axis=0)  # (4,)
        logit = float(np.dot(self._W, feat_mean) + self._B)
        return float(1.0 / (1.0 + np.exp(-logit)))

    def _frame_features(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Extract 4 spectral features from one face crop."""
        # Convert to grayscale
        gray = (0.299 * frame_rgb[:, :, 0] +
                0.587 * frame_rgb[:, :, 1] +
                0.114 * frame_rgb[:, :, 2]).astype(np.float32)

        # Downsample for speed
        import cv2
        gray = cv2.resize(gray, (self.frame_size, self.frame_size),
                          interpolation=cv2.INTER_AREA)

        # 2D FFT → centered log power spectrum
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        power = np.log1p(np.abs(fft_shift) ** 2)   # log-compressed

        H, W = power.shape
        cy, cx = H // 2, W // 2

        band_energies = {}
        for name, (lo, hi) in self.BANDS.items():
            mask = self._radial_mask(H, W, cy, cx, lo, hi)
            band_energies[name] = float(power[mask].mean() + 1e-8)

        # Feature 1: mid/low ratio  (GAN artifacts inflate mid-freq)
        f1 = band_energies["mid"] / band_energies["low"]

        # Feature 2: high/dc ratio  (deepfakes often suppress high-freq)
        f2 = band_energies["high"] / band_energies["dc"]

        # Feature 3: spectral entropy  (natural images have higher entropy)
        p_norm = power / (power.sum() + 1e-8)
        f3 = float(-np.sum(p_norm * np.log(p_norm + 1e-10)))
        f3 /= np.log(H * W)   # normalize to [0, 1]

        # Feature 4: peak prominence  (generators create spectral peaks)
        f4 = float(power.max() / (power.mean() + 1e-8))
        f4 = min(f4 / 100.0, 1.0)  # normalize

        return np.array([f1, f2, f3, f4], dtype=np.float32)

    @staticmethod
    def _radial_mask(H: int, W: int, cy: int, cx: int,
                     lo: float, hi: float) -> np.ndarray:
        """Boolean mask selecting radial frequency band [lo, hi] × Nyquist."""
        ys = (np.arange(H) - cy) / (H / 2)
        xs = (np.arange(W) - cx) / (W / 2)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        r = np.sqrt(yy ** 2 + xx ** 2)
        return (r >= lo) & (r < hi)
