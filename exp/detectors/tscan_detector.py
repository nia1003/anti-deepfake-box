"""
TS-CAN: Temporal Shift Convolutional Attention Network for rPPG-based deepfake detection.

Architecture from: Liu et al. (2020). "Multi-Task Temporal Shift Attention Networks
for On-Device Contactless Vitals Measurement." NeurIPS 2020.

Pipeline:
  1. Resize face crops to tscan_img_size × tscan_img_size
  2. Build motion (frame diff) and appearance (raw frames) inputs
  3. Slide TS-CAN window (frame_depth frames) with stride 1
  4. Average predicted rPPG signal across windows
  5. Compute SNR; map to fake score (same as POS-based RPPGDetector)

Fallback: if checkpoint absent or import fails, degrades to POS algorithm
(identical to the project's existing RPPGDetector).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detectors.base_detector import BaseDetector
from detectors.rppg_detector import _pos_wang, compute_ppg_snr, snr_to_fake_score
from preprocessing.face_extractor import FaceTrack

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


# ---------------------------------------------------------------------------
# TS-CAN model
# ---------------------------------------------------------------------------

class _TemporalAttention(nn.Module):
    """Channel-wise temporal attention gate (from TS-CAN)."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.fc1 = nn.Linear(in_ch, in_ch // 2)
        self.fc2 = nn.Linear(in_ch // 2, in_ch)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (B, C, H, W)
        gap = x.mean(dim=[2, 3])            # (B, C)
        att = torch.sigmoid(self.fc2(F.relu(self.fc1(gap))))  # (B, C)
        return x * att.unsqueeze(-1).unsqueeze(-1)


class TSCAN(nn.Module):
    """
    TS-CAN network.

    Input:
        motion:     (B, C*frame_depth, H, W) — normalised frame differences
        appearance: (B, C*frame_depth, H, W) — normalised raw frames

    Output:
        rppg: (B, frame_depth) — predicted rPPG signal
    """

    def __init__(
        self,
        frame_depth: int = 20,
        img_size: int = 36,
        nb_filters1: int = 32,
        nb_filters2: int = 64,
        kernel_size: int = 3,
        nb_dense: int = 128,
        dropout1: float = 0.25,
        dropout2: float = 0.5,
    ):
        super().__init__()
        in_ch = 3 * frame_depth

        # Motion branch
        self.motion_conv1 = nn.Conv2d(in_ch, nb_filters1, kernel_size, padding=1)
        self.motion_bn1 = nn.BatchNorm2d(nb_filters1)
        self.motion_conv2 = nn.Conv2d(nb_filters1, nb_filters1, kernel_size, padding=1)
        self.motion_bn2 = nn.BatchNorm2d(nb_filters1)
        self.motion_pool = nn.AvgPool2d(2)
        self.motion_drop1 = nn.Dropout2d(dropout1)

        # Appearance branch
        self.app_conv1 = nn.Conv2d(in_ch, nb_filters1, kernel_size, padding=1)
        self.app_bn1 = nn.BatchNorm2d(nb_filters1)
        self.app_conv2 = nn.Conv2d(nb_filters1, nb_filters1, kernel_size, padding=1)
        self.app_bn2 = nn.BatchNorm2d(nb_filters1)
        self.app_pool = nn.AvgPool2d(2)
        self.app_drop1 = nn.Dropout2d(dropout1)

        # Temporal attention (conditioned on appearance)
        self.att = _TemporalAttention(nb_filters1)

        # Merged conv
        self.merge_conv1 = nn.Conv2d(nb_filters1, nb_filters2, kernel_size, padding=1)
        self.merge_bn1 = nn.BatchNorm2d(nb_filters2)
        self.merge_conv2 = nn.Conv2d(nb_filters2, nb_filters2, kernel_size, padding=1)
        self.merge_bn2 = nn.BatchNorm2d(nb_filters2)
        self.merge_pool = nn.AvgPool2d(2)
        self.merge_drop = nn.Dropout2d(dropout2)

        pooled_size = img_size // 4
        flat_dim = nb_filters2 * pooled_size * pooled_size
        self.fc1 = nn.Linear(flat_dim, nb_dense)
        self.fc_drop = nn.Dropout(dropout2)
        self.fc2 = nn.Linear(nb_dense, frame_depth)

    def forward(self, motion: "torch.Tensor", appearance: "torch.Tensor") -> "torch.Tensor":
        m = self.motion_drop1(self.motion_pool(
            F.relu(self.motion_bn2(self.motion_conv2(
                F.relu(self.motion_bn1(self.motion_conv1(motion))))))))

        a = self.app_drop1(self.app_pool(
            F.relu(self.app_bn2(self.app_conv2(
                F.relu(self.app_bn1(self.app_conv1(appearance))))))))

        # Attention gate: appearance weights motion features
        gate = torch.sigmoid(self.att(a))
        fused = m * gate

        x = self.merge_drop(self.merge_pool(
            F.relu(self.merge_bn2(self.merge_conv2(
                F.relu(self.merge_bn1(self.merge_conv1(fused))))))))

        x = x.flatten(1)
        x = self.fc_drop(F.relu(self.fc1(x)))
        return self.fc2(x)   # (B, frame_depth) rPPG per frame


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _resize_crops(crops: np.ndarray, size: int) -> np.ndarray:
    """(T, H, W, 3) uint8 → (T, size, size, 3) float32 in [0, 1]."""
    from PIL import Image
    out = np.zeros((len(crops), size, size, 3), dtype=np.float32)
    for i, c in enumerate(crops):
        img = Image.fromarray(c).resize((size, size), Image.BILINEAR)
        out[i] = np.array(img, dtype=np.float32) / 255.0
    return out


def _build_motion(frames: np.ndarray) -> np.ndarray:
    """
    Compute normalised frame differences for motion input.

    frames: (T, H, W, 3) float32 in [0,1]
    returns: (T-1, H, W, 3) float32 — diff / (mean_brightness + eps)
    """
    diff = frames[1:] - frames[:-1]  # (T-1, H, W, 3)
    mean_brightness = frames[:-1].mean(axis=(1, 2, 3), keepdims=True) + 1e-6
    return diff / mean_brightness


def _tscan_predict(
    model: "TSCAN",
    frames: np.ndarray,
    frame_depth: int,
    device: str,
) -> np.ndarray:
    """
    Slide TS-CAN over frames; return per-frame rPPG signal of length T-1.

    frames: (T, H, W, 3) float32 in [0, 1]
    """
    import torch

    motion = _build_motion(frames)    # (T-1, H, W, 3)
    T = len(motion)
    rppg = np.zeros(T, dtype=np.float64)
    count = np.zeros(T, dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for start in range(0, T - frame_depth + 1):
            end = start + frame_depth
            m_chunk = motion[start:end]  # (fd, H, W, 3)
            a_chunk = frames[start:end]  # (fd, H, W, 3)

            # (fd, H, W, 3) → (1, 3*fd, H, W)
            H, W = m_chunk.shape[1:3]
            m_t = torch.from_numpy(
                m_chunk.transpose(3, 0, 1, 2).reshape(1, -1, H, W)
            ).to(device)
            a_t = torch.from_numpy(
                a_chunk.transpose(3, 0, 1, 2).reshape(1, -1, H, W)
            ).to(device)

            pred = model(m_t, a_t).cpu().numpy()[0]  # (frame_depth,)
            rppg[start:end] += pred
            count[start:end] += 1

    mask = count > 0
    rppg[mask] /= count[mask]
    return rppg


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class TSCANDetector(BaseDetector):
    """
    TS-CAN rPPG detector for deepfake identification.

    If checkpoint is available and torch is installed, uses the neural TS-CAN
    model to estimate the rPPG signal. Otherwise falls back to the POS
    algorithm (Wang 2017), producing equivalent SNR-based fake scores.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.pretrained: str = config.get("pretrained", "")
        self.frame_depth: int = int(config.get("tscan_frame_depth", 20))
        self.img_size: int = int(config.get("tscan_img_size", 36))
        self.snr_threshold: float = float(config.get("snr_threshold", 1.5))
        self.snr_scale: float = float(config.get("snr_scale", 1.0))
        self._use_tscan: bool = False
        self._model = None

    def load(self) -> None:
        if not _TORCH_OK:
            return  # fall back to POS

        ckpt_path = Path(self.pretrained)
        if not ckpt_path.exists():
            return  # fall back to POS

        try:
            model = TSCAN(frame_depth=self.frame_depth, img_size=self.img_size)
            state = torch.load(ckpt_path, map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            elif isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
            model.eval()
            self._model = model.to(self.device)
            self._use_tscan = True
        except Exception as exc:
            print(f"[TSCANDetector] Failed to load checkpoint ({exc}); using POS fallback.")

    # ------------------------------------------------------------------
    # rPPG estimation
    # ------------------------------------------------------------------

    def _estimate_rppg_tscan(self, face_track: FaceTrack) -> np.ndarray:
        """Use TS-CAN neural model to estimate rPPG signal."""
        frames = _resize_crops(face_track.crops_128, self.img_size)
        return _tscan_predict(self._model, frames, self.frame_depth, self.device)

    def _estimate_rppg_pos(self, face_track: FaceTrack) -> np.ndarray:
        """Fall back to POS algorithm (Wang 2017)."""
        return _pos_wang(face_track.crops_128, fps=face_track.fps)

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def _detect_impl(
        self,
        face_track: FaceTrack,
        audio_path: Optional[str] = None,
    ) -> Optional[float]:
        if face_track is None or face_track.T == 0:
            return None

        if self._use_tscan:
            bvp = self._estimate_rppg_tscan(face_track)
            method = "TS-CAN"
        else:
            bvp = self._estimate_rppg_pos(face_track)
            method = "POS"

        snr = compute_ppg_snr(bvp, fps=face_track.fps)
        score = snr_to_fake_score(snr, self.snr_threshold, self.snr_scale)
        return score

    def get_ppg_and_snr(self, face_track: FaceTrack) -> dict:
        """Debug helper: return BVP waveform, SNR, and fake score."""
        if not self._loaded:
            self.load()
            self._loaded = True

        if self._use_tscan:
            bvp = self._estimate_rppg_tscan(face_track)
            method = "TS-CAN"
        else:
            bvp = self._estimate_rppg_pos(face_track)
            method = "POS"

        snr = compute_ppg_snr(bvp, fps=face_track.fps)
        return {
            "ppg": bvp,
            "snr_db": snr,
            "fake_score": snr_to_fake_score(snr, self.snr_threshold, self.snr_scale),
            "method": method,
            "num_frames": face_track.T,
        }
