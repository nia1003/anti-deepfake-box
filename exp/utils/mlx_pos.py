"""
MLX-accelerated POS (Plane-Orthogonal-to-Skin) rPPG algorithm.

On Apple Silicon, MLX dispatches the per-window matrix operations to the
Metal GPU, reducing wall-clock time compared to pure NumPy—especially for
longer clips.  The sliding-window accumulation loop stays in Python; only
the inner linear-algebra kernel (projection, normalisation, alpha weighting)
runs on Metal.

Falls back to the existing NumPy/SciPy implementation when MLX is absent.

Reference: Wang et al. (2017). Algorithmic principles of remote PPG.
           IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491.
"""

from __future__ import annotations

import math

import numpy as np

# _P projection matrix (row vectors)
_P_NP = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]], dtype=np.float64)


def _pos_mlx(frames: np.ndarray, fps: float) -> np.ndarray:
    """
    POS algorithm using MLX for inner window operations.

    Parameters
    ----------
    frames : (T, H, W, 3) uint8 RGB
    fps    : frames per second

    Returns
    -------
    bvp : (T,) float64  bandpass-filtered blood volume pulse
    """
    import mlx.core as mx
    from scipy import signal as sp_signal

    T = len(frames)
    l = max(2, math.ceil(1.6 * fps))

    # Spatial mean per frame → (T, 3) on Metal
    rgb_np = frames.astype(np.float32).reshape(T, -1, 3).mean(axis=1)
    rgb = mx.array(rgb_np)                          # (T, 3) — lives on Metal

    P = mx.array(_P_NP.astype(np.float32))         # (2, 3)

    H = np.zeros(T, dtype=np.float64)

    for n in range(l, T):
        m = n - l
        chunk = rgb[m:n]                            # (l, 3) slice — lazy
        mean_c = chunk.mean(axis=0) + 1e-6         # (3,)
        Cn = (chunk / mean_c).T                     # (3, l)
        S = P @ Cn                                  # (2, l)

        mx.eval(S)                                  # flush Metal command buffer
        S_np = np.array(S, dtype=np.float64)

        alpha = np.std(S_np[0]) / (np.std(S_np[1]) + 1e-9)
        h = S_np[0] + alpha * S_np[1]
        h -= h.mean()
        H[m:n] += h

    # Bandpass 0.75–3 Hz (physiological HR)
    nyq = fps / 2.0
    b, a = sp_signal.butter(
        1, [0.75 / nyq, min(3.0 / nyq, 0.99)], btype="bandpass"
    )
    return sp_signal.filtfilt(b, a, H)


def _pos_numpy(frames: np.ndarray, fps: float) -> np.ndarray:
    """Pure NumPy/SciPy POS — used when MLX is unavailable."""
    from scipy import signal as sp_signal

    T = len(frames)
    l = max(2, math.ceil(1.6 * fps))

    rgb = frames.astype(np.float64).reshape(T, -1, 3).mean(axis=1)
    H = np.zeros(T, dtype=np.float64)

    for n in range(l, T):
        m = n - l
        chunk = rgb[m:n]
        mean_c = chunk.mean(axis=0) + 1e-6
        Cn = (chunk / mean_c).T
        S = _P_NP @ Cn
        alpha = np.std(S[0]) / (np.std(S[1]) + 1e-9)
        h = S[0] + alpha * S[1]
        h -= h.mean()
        H[m:n] += h

    nyq = fps / 2.0
    b, a = sp_signal.butter(
        1, [0.75 / nyq, min(3.0 / nyq, 0.99)], btype="bandpass"
    )
    return sp_signal.filtfilt(b, a, H)


def pos_wang(frames: np.ndarray, fps: float) -> np.ndarray:
    """
    POS rPPG estimation: uses MLX on Apple Silicon, NumPy elsewhere.

    Parameters
    ----------
    frames : (T, H, W, 3) uint8 RGB face crops
    fps    : capture frame rate

    Returns
    -------
    bvp : (T,) float64 bandpass-filtered BVP signal
    """
    try:
        import mlx.core  # noqa: F401
        return _pos_mlx(frames, fps)
    except ImportError:
        return _pos_numpy(frames, fps)
