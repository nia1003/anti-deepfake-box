"""
Anti-Deepfake-Box: Visual Detector adapter for DeepfakeBench.

Wraps the VisualDetector (XceptionNet) to be compatible with
DeepfakeBench's AbstractDetector interface.

Install: copy this file to
    <deepfakebench>/training/detectors/adb_visual_detector.py
and add the following import to
    <deepfakebench>/training/detectors/__init__.py:
        from .adb_visual_detector import ADBVisualDetector
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------ #
#  Try importing DeepfakeBench infrastructure                          #
# ------------------------------------------------------------------ #
try:
    from detectors import DETECTOR
    from .base_detector import AbstractDetector
    from loss import LOSSFUNC
    DFB_AVAILABLE = True
except ImportError:
    # Standalone mode (testing without DFB installation)
    DFB_AVAILABLE = False
    class AbstractDetector(nn.Module):
        pass
    def register_module(module_name):
        def decorator(cls): return cls
        return decorator
    DETECTOR = type("DETECTOR", (), {"register_module": staticmethod(register_module)})()

# ADB path
_ADB_ROOT = Path(__file__).parent.parent
if str(_ADB_ROOT) not in sys.path:
    sys.path.insert(0, str(_ADB_ROOT))

from detectors.visual_detector import VisualDetector as ADBVisualDetector_impl
from preprocessing.face_extractor import UnifiedFaceExtractor, FaceTrack


def _dfb_batch_to_face_track(image_tensor: torch.Tensor, fps: float = 25.0) -> FaceTrack:
    """
    Convert DeepfakeBench image batch (B, 3, H, W) or (B, T, 3, H, W)
    to a FaceTrack with aligned_256.

    DFB preprocessing already crops and resizes faces; we use them directly
    as aligned_256 without re-running InsightFace.
    """
    import numpy as np

    if image_tensor.dim() == 4:
        # (B, 3, H, W) → treat as T frames, batch=1
        x = image_tensor  # (T, 3, H, W)
    elif image_tensor.dim() == 5:
        # (1, T, 3, H, W) → (T, 3, H, W)
        x = image_tensor.squeeze(0)
    else:
        raise ValueError(f"Unexpected image tensor dim: {image_tensor.dim()}")

    T = x.shape[0]
    H, W = x.shape[2], x.shape[3]

    # Denormalise from [-1,1] or [0,1] to uint8
    x_np = x.detach().cpu().float().numpy()
    if x_np.min() < -0.1:  # normalised with [-1,1]
        x_np = (x_np * 0.5 + 0.5)
    x_np = (x_np * 255).clip(0, 255).astype("uint8")  # (T, 3, H, W)
    x_np = x_np.transpose(0, 2, 3, 1)  # (T, H, W, 3) RGB

    # Resize to 256x256 for FaceTrack base
    import cv2
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


@DETECTOR.register_module(module_name='adb_visual')
class ADBVisualDetector(AbstractDetector):
    """
    DeepfakeBench-compatible wrapper for Anti-Deepfake-Box VisualDetector.

    Config YAML fields:
        adb_config_path: path to anti-deepfake-box configs/default.yaml
        visual_pretrained: path to XceptionNet FF++ checkpoint
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Build ADB detector config from DFB config
        adb_cfg = {
            "device": config.get("device", "cuda"),
            "visual_pretrained": config.get("visual_pretrained", ""),
            "visual_max_frames": config.get("frame_num", {}).get("test", 32)
                if isinstance(config.get("frame_num"), dict)
                else config.get("frame_num", 32),
            "visual_batch_size": config.get("test_batchSize", 16),
        }

        self.adb_detector = ADBVisualDetector_impl(adb_cfg)
        self.loss_func = self.build_loss(config)

        self.prob, self.label = [], []
        self.video_names = []
        self.correct, self.total = 0, 0

    def build_backbone(self, config):
        # Backbone is encapsulated in ADBVisualDetector_impl
        return None

    def build_loss(self, config):
        try:
            loss_class = LOSSFUNC[config.get("loss_func", "cross_entropy")]
            return loss_class()
        except Exception:
            return nn.CrossEntropyLoss()

    def features(self, data_dict: dict) -> torch.Tensor:
        """Extract visual features from DFB image batch."""
        face_track = _dfb_batch_to_face_track(data_dict["image"])
        if not self.adb_detector._loaded:
            self.adb_detector.load()
            self.adb_detector._loaded = True

        # Get intermediate features from XceptionNet
        crops = torch.from_numpy(
            face_track.to_float32_chw(299).copy()
        ).to(self.config.get("device", "cuda"))

        if self.adb_detector.model is not None:
            try:
                with torch.no_grad():
                    feats = self.adb_detector.model.features(crops)
                return feats
            except AttributeError:
                pass
        return torch.zeros(crops.size(0), 2048, device=crops.device)

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        if self.adb_detector.model is not None:
            try:
                return self.adb_detector.model.classifier(features)
            except AttributeError:
                pass
        # Fallback linear head
        B = features.shape[0]
        return torch.zeros(B, 2, device=features.device)

    def forward(self, data_dict: dict, inference: bool = False) -> dict:
        face_track = _dfb_batch_to_face_track(data_dict["image"])
        if not self.adb_detector._loaded:
            self.adb_detector.load()
            self.adb_detector._loaded = True

        # Per-frame fake scores (B,)
        score = self.adb_detector._detect_impl(face_track)
        if score is None:
            score = 0.5

        # DFB expects (B, 2) logits
        B = data_dict["image"].shape[0]
        device = data_dict["image"].device
        prob_tensor = torch.full((B,), score, dtype=torch.float32, device=device)
        fake_logit = torch.log(prob_tensor.clamp(1e-7, 1 - 1e-7))
        real_logit = torch.log((1 - prob_tensor).clamp(1e-7, 1 - 1e-7))
        logits = torch.stack([real_logit, fake_logit], dim=1)

        feats = self.features(data_dict)
        return {"cls": logits, "prob": prob_tensor, "feat": feats}

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict["label"]
        pred = pred_dict["cls"]
        loss = self.loss_func(pred, label)
        return {"overall": loss, "cls": loss}

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
        import numpy as np

        label = data_dict["label"].detach().cpu().numpy()
        prob = pred_dict["prob"].detach().cpu().numpy()
        pred_bin = (prob >= 0.5).astype(int)

        try:
            auc = float(roc_auc_score(label, prob))
        except Exception:
            auc = 0.0
        acc = float(accuracy_score(label, pred_bin))

        return {"acc": acc, "auc": auc, "eer": 0.0, "ap": 0.0}
