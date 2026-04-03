"""
Anti-Deepfake-Box: rPPG Detector adapter for DeepfakeBench.

Wraps the RPPGDetector (PhysNet + SNR scoring) for DFB compatibility.

Install: copy to <deepfakebench>/training/detectors/adb_rppg_detector.py
Add import to <deepfakebench>/training/detectors/__init__.py:
    from .adb_rppg_detector import ADBRPPGDetector

Note: DFB config should set video_mode: true, frame_num: 180 for this detector.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn

try:
    from detectors import DETECTOR
    from .base_detector import AbstractDetector
    from loss import LOSSFUNC
    DFB_AVAILABLE = True
except ImportError:
    DFB_AVAILABLE = False
    class AbstractDetector(nn.Module):
        pass
    def register_module(module_name):
        def decorator(cls): return cls
        return decorator
    DETECTOR = type("DETECTOR", (), {"register_module": staticmethod(register_module)})()

_ADB_ROOT = Path(__file__).parent.parent
if str(_ADB_ROOT) not in sys.path:
    sys.path.insert(0, str(_ADB_ROOT))

from detectors.rppg_detector import RPPGDetector as ADBRPPGDetector_impl, compute_ppg_snr, snr_to_fake_score
from preprocessing.face_extractor import FaceTrack


def _dfb_video_batch_to_face_track(image_tensor: torch.Tensor, fps: float = 25.0) -> FaceTrack:
    """
    Convert DFB video batch (B, T, 3, H, W) or (B, 3, H, W)
    to FaceTrack with crops_128 usable by PhysNet.
    """
    import cv2

    if image_tensor.dim() == 5:
        x = image_tensor[0]   # (T, 3, H, W) — take first sample
    elif image_tensor.dim() == 4:
        x = image_tensor      # (T, 3, H, W)
    else:
        raise ValueError(f"Unexpected tensor dim: {image_tensor.dim()}")

    T = x.shape[0]
    x_np = x.detach().cpu().float().numpy()
    if x_np.min() < -0.1:
        x_np = x_np * 0.5 + 0.5
    x_np = (x_np * 255).clip(0, 255).astype("uint8")
    x_np = x_np.transpose(0, 2, 3, 1)  # (T, H, W, 3) RGB

    # Resize to 256 (base) then 128 derived on demand
    H, W = x_np.shape[1], x_np.shape[2]
    if H != 256 or W != 256:
        resized = np.stack([cv2.resize(f, (256, 256)) for f in x_np])
    else:
        resized = x_np

    return FaceTrack(
        frame_indices=np.arange(T),
        bboxes=np.zeros((T, 4), dtype=np.float32),
        landmarks=np.zeros((T, 5, 2), dtype=np.float32),
        aligned_256=resized,
        fps=fps,
        video_path="",
    )


@DETECTOR.register_module(module_name='adb_rppg')
class ADBRPPGDetector(AbstractDetector):
    """
    DeepfakeBench-compatible wrapper for Anti-Deepfake-Box RPPGDetector.

    DFB config YAML should include:
        video_mode: true
        frame_num:
          train: 180
          test: 180
        rppg_pretrained: path/to/physnet.pth
        snr_threshold: 1.5
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        adb_cfg = {
            "device": config.get("device", "cuda"),
            "rppg_pretrained": config.get("rppg_pretrained", ""),
            "snr_threshold": float(config.get("snr_threshold", 1.5)),
            "snr_scale": float(config.get("snr_scale", 1.0)),
        }
        self.adb_detector = ADBRPPGDetector_impl(adb_cfg)
        self.loss_func = self.build_loss(config)
        self.prob, self.label = [], []
        self.video_names = []
        self.correct, self.total = 0, 0

    def build_backbone(self, config):
        return None

    def build_loss(self, config):
        try:
            loss_class = LOSSFUNC[config.get("loss_func", "cross_entropy")]
            return loss_class()
        except Exception:
            return nn.CrossEntropyLoss()

    def features(self, data_dict: dict) -> torch.Tensor:
        """Return SNR value as a 1-D feature tensor (B, 1)."""
        face_track = _dfb_video_batch_to_face_track(data_dict["image"])
        if not self.adb_detector._loaded:
            self.adb_detector.load()
            self.adb_detector._loaded = True

        info = self.adb_detector.get_ppg_and_snr(face_track)
        snr = info["snr_db"]
        B = data_dict["image"].shape[0]
        device = data_dict["image"].device
        return torch.full((B, 1), snr, dtype=torch.float32, device=device)

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        """Map SNR feature to [real_logit, fake_logit]."""
        # features: (B, 1) containing SNR in dB
        snr = features[:, 0]
        threshold = float(self.config.get("snr_threshold", 1.5))
        scale = float(self.config.get("snr_scale", 1.0))
        # fake_prob via sigmoid
        fake_prob = torch.sigmoid(-(snr - threshold) * scale)
        real_prob = 1.0 - fake_prob
        eps = 1e-7
        logits = torch.stack([
            torch.log(real_prob.clamp(eps, 1 - eps)),
            torch.log(fake_prob.clamp(eps, 1 - eps)),
        ], dim=1)
        return logits

    def forward(self, data_dict: dict, inference: bool = False) -> dict:
        face_track = _dfb_video_batch_to_face_track(data_dict["image"])
        if not self.adb_detector._loaded:
            self.adb_detector.load()
            self.adb_detector._loaded = True

        score = self.adb_detector._detect_impl(face_track)
        score = score if score is not None else 0.5

        B = data_dict["image"].shape[0]
        device = data_dict["image"].device

        prob_tensor = torch.full((B,), score, dtype=torch.float32, device=device)
        eps = 1e-7
        logits = torch.stack([
            torch.log((1 - prob_tensor).clamp(eps, 1 - eps)),
            torch.log(prob_tensor.clamp(eps, 1 - eps)),
        ], dim=1)

        feats = self.features(data_dict)
        return {"cls": logits, "prob": prob_tensor, "feat": feats}

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict["label"]
        pred = pred_dict["cls"]
        loss = self.loss_func(pred, label)
        return {"overall": loss, "cls": loss}

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        from sklearn.metrics import roc_auc_score, accuracy_score
        label = data_dict["label"].detach().cpu().numpy()
        prob = pred_dict["prob"].detach().cpu().numpy()
        pred_bin = (prob >= 0.5).astype(int)
        try:
            auc = float(roc_auc_score(label, prob))
        except Exception:
            auc = 0.0
        acc = float(accuracy_score(label, pred_bin))
        return {"acc": acc, "auc": auc, "eer": 0.0, "ap": 0.0}
