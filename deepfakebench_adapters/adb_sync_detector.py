"""
Anti-Deepfake-Box: Sync Detector adapter for DeepfakeBench.

Wraps the SyncDetector (LatentSync SyncNet) for DFB compatibility.

Note: DFB's standard preprocessing does NOT extract audio.
This adapter reads the original video_path to extract audio on-the-fly.
Set 'video_mode: true' in the detector config.

Install: copy to <deepfakebench>/training/detectors/adb_sync_detector.py
Add import to __init__.py: from .adb_sync_detector import ADBSyncDetector
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

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

from detectors.sync_detector import SyncDetector as ADBSyncDetector_impl
from preprocessing.audio_extractor import AudioExtractor
from preprocessing.face_extractor import FaceTrack

from deepfakebench_adapters.adb_visual_detector import _dfb_batch_to_face_track


@DETECTOR.register_module(module_name='adb_sync')
class ADBSyncDetector(AbstractDetector):
    """
    DeepfakeBench-compatible wrapper for Anti-Deepfake-Box SyncDetector.

    Requires 'video_path' key in data_dict for audio extraction.
    Falls back to 0.5 (uncertain) when audio is unavailable.

    DFB config YAML:
        video_mode: true
        frame_num:
          train: 25
          test: 25
        syncnet_path: path/to/syncnet.pth
        whisper_model: "tiny"
        whisper_device: "cpu"
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        adb_cfg = {
            "device": config.get("device", "cuda"),
            "syncnet_path": config.get("syncnet_path", ""),
            "whisper_model": config.get("whisper_model", "tiny"),
            "whisper_device": config.get("whisper_device", "cpu"),
        }
        self.adb_detector = ADBSyncDetector_impl(adb_cfg)
        self.audio_extractor = AudioExtractor()
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

    def _get_audio_path(self, data_dict: dict) -> str | None:
        """Extract audio from video_path if provided in data_dict."""
        video_path = data_dict.get("video_path", None)
        if video_path is None:
            return None
        if isinstance(video_path, list):
            video_path = video_path[0]
        return self.audio_extractor.extract_to_temp(str(video_path))

    def features(self, data_dict: dict) -> torch.Tensor:
        """Return sync confidence as feature (B, 1)."""
        face_track = _dfb_batch_to_face_track(data_dict["image"])
        if not self.adb_detector._loaded:
            self.adb_detector.load()
            self.adb_detector._loaded = True

        audio_path = self._get_audio_path(data_dict)
        score = self.adb_detector._detect_impl(face_track, audio_path)
        if audio_path and Path(audio_path).exists():
            try:
                os.unlink(audio_path)
            except OSError:
                pass

        score = score if score is not None else 0.5
        B = data_dict["image"].shape[0]
        device = data_dict["image"].device
        return torch.full((B, 1), 1.0 - score, dtype=torch.float32, device=device)

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        fake_prob = features[:, 0].clamp(0, 1)
        real_prob = 1.0 - fake_prob
        eps = 1e-7
        return torch.stack([
            torch.log(real_prob.clamp(eps, 1 - eps)),
            torch.log(fake_prob.clamp(eps, 1 - eps)),
        ], dim=1)

    def forward(self, data_dict: dict, inference: bool = False) -> dict:
        face_track = _dfb_batch_to_face_track(data_dict["image"])
        if not self.adb_detector._loaded:
            self.adb_detector.load()
            self.adb_detector._loaded = True

        audio_path = self._get_audio_path(data_dict)
        score = self.adb_detector._detect_impl(face_track, audio_path)
        if audio_path and Path(audio_path).exists():
            try:
                os.unlink(audio_path)
            except OSError:
                pass

        score = score if score is not None else 0.5
        B = data_dict["image"].shape[0]
        device = data_dict["image"].device
        prob_tensor = torch.full((B,), score, dtype=torch.float32, device=device)
        eps = 1e-7
        logits = torch.stack([
            torch.log((1 - prob_tensor).clamp(eps, 1 - eps)),
            torch.log(prob_tensor.clamp(eps, 1 - eps)),
        ], dim=1)
        return {"cls": logits, "prob": prob_tensor, "feat": prob_tensor.unsqueeze(1)}

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
