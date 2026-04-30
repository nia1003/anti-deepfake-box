"""
Device auto-detection and backend availability checks.

Priority order for "auto":
  1. CUDA   — NVIDIA GPU (Linux / Windows)
  2. MPS    — Apple Silicon Metal (macOS >= 12.3, PyTorch >= 2.0)
  3. CPU    — fallback

MLX is Apple's own ML framework (separate from PyTorch MPS).
It is used for signal-processing operations (rPPG / POS) on Apple Silicon.
PyTorch models use MPS; signal ops use MLX.
"""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def is_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@functools.lru_cache(maxsize=1)
def is_mps_available() -> bool:
    """True when running on Apple Silicon with PyTorch MPS support."""
    try:
        import torch
        return torch.backends.mps.is_available()
    except (ImportError, AttributeError):
        return False


@functools.lru_cache(maxsize=1)
def is_mlx_available() -> bool:
    """True when Apple MLX is installed (pip install mlx)."""
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


def get_device(preferred: str = "auto") -> str:
    """
    Resolve a device string.

    "auto" → cuda > mps > cpu
    Any other value is returned as-is after basic validation.
    """
    if preferred != "auto":
        return preferred

    if is_cuda_available():
        return "cuda"
    if is_mps_available():
        return "mps"
    return "cpu"


def describe_backends() -> str:
    """Human-readable summary of available compute backends."""
    parts = []
    parts.append(f"CUDA  : {'yes' if is_cuda_available() else 'no'}")
    parts.append(f"MPS   : {'yes' if is_mps_available() else 'no'}")
    parts.append(f"MLX   : {'yes' if is_mlx_available() else 'no'}")
    parts.append(f"Active: {get_device('auto')}")
    return "\n".join(parts)
