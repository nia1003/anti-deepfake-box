"""
Anti-Deepfake-Box: Ensemble Detector adapter for DeepfakeBench.

Combines all three modalities (visual + rPPG + sync) into a single
DeepfakeBench-compatible detector for alignment analysis.

Install: copy to <deepfakebench>/training/detectors/adb_ensemble_detector.py
Add import to __init__.py: from .adb_ensemble_detector import ADBEnsembleDetector
"""

from __future__ import annotations

import os
import sys
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

from detectors.visual_detector import VisualDetector as ADBVisualDetector_impl
from detectors.rppg_detector import RPPGDetector as ADBRPPGDetector_impl
from detectors.sync_detector import SyncDetector as ADBSyncDetector_impl
from fusion.weighted_ensemble import WeightedEnsemble
from preprocessing.audio_extractor import AudioExtractor
from deepfakebench_adapters.adb_visual_detector import _dfb_batch_to_face_track


@DETECTOR.register_module(module_name='adb_ensemble')
class ADBEnsembleDetector(AbstractDetector):
    """
    DeepfakeBench-compatible three-modality ensemble detector.

    Scores from all three detectors are fused via WeightedEnsemble.
    Gracefully handles missing audio (sync=None → excluded from fusion).

    DFB config YAML:
        visual_pretrained: path/to/xception.pth
        rppg_pretrained: path/to/physnet.pth
        syncnet_path: path/to/syncnet.pth
        snr_threshold: 1.5
        fusion_weights:
          visual: 0.50
          rppg:   0.25
          sync:   0.25
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        device = config.get("device", "cuda")

        visual_cfg = {
            "device": device,
            "visual_pretrained": config.get("visual_pretrained", ""),
            "visual_max_frames": 32,
            "visual_batch_size": 16,
        }
        rppg_cfg = {
            "device": device,
            "rppg_pretrained": config.get("rppg_pretrained", ""),
            "snr_threshold": float(config.get("snr_threshold", 1.5)),
            "snr_scale": float(config.get("snr_scale", 1.0)),
        }
        sync_cfg = {
            "device": device,
            "syncnet_path": config.get("syncnet_path", ""),
            "whisper_model": config.get("whisper_model", "tiny"),
            "whisper_device": config.get("whisper_device", "cpu"),
        }

        self.visual_det = ADBVisualDetector_impl(visual_cfg)
        self.rppg_det   = ADBRPPGDetector_impl(rppg_cfg)
        self.sync_det   = ADBSyncDetector_impl(sync_cfg)
        self.audio_extractor = AudioExtractor()

        fw = config.get("fusion_weights", {})
        fusion_config = {
            "fusion": {
                "weights": {
                    "visual": float(fw.get("visual", 0.50)),
                    "rppg":   float(fw.get("rppg",   0.25)),
                    "sync":   float(fw.get("sync",   0.25)),
                },
                "threshold": float(config.get("fusion_threshold", 0.50)),
            }
        }
        self.fuser = WeightedEnsemble(fusion_config)
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

    def _ensure_loaded(self):
        for det in [self.visual_det, self.rppg_det, self.sync_det]:
            if not det._loaded:
                det.load()
                det._loaded = True

    def features(self, data_dict: dict) -> torch.Tensor:
        """Return concatenated [visual_score, rppg_score, sync_score] as features."""
        self._ensure_loaded()
        face_track = _dfb_batch_to_face_track(data_dict["image"])

        audio_path = None
        video_path = data_dict.get("video_path")
        if video_path:
            if isinstance(video_path, list):
                video_path = video_path[0]
            audio_path = self.audio_extractor.extract_to_temp(str(video_path))

        s_v = self.visual_det._detect_impl(face_track)
        s_r = self.rppg_det._detect_impl(face_track)
        s_s = self.sync_det._detect_impl(face_track, audio_path)

        if audio_path and Path(audio_path).exists():
            try:
                os.unlink(audio_path)
            except OSError:
                pass

        B = data_dict["image"].shape[0]
        device = data_dict["image"].device
        feat_vec = torch.tensor([
            s_v if s_v is not None else 0.5,
            s_r if s_r is not None else 0.5,
            s_s if s_s is not None else 0.5,
        ], dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1)

        return feat_vec

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, 3) = [visual, rppg, sync]
        # weighted average → logits
        w = torch.tensor(
            [self.fuser.weights["visual"], self.fuser.weights["rppg"], self.fuser.weights["sync"]],
            device=features.device
        )
        fake_prob = (features * w).sum(dim=1) / w.sum()
        real_prob = 1.0 - fake_prob
        eps = 1e-7
        return torch.stack([
            torch.log(real_prob.clamp(eps, 1 - eps)),
            torch.log(fake_prob.clamp(eps, 1 - eps)),
        ], dim=1)

    def forward(self, data_dict: dict, inference: bool = False) -> dict:
        feats = self.features(data_dict)
        # feats: (B, 3)
        logits = self.classifier(feats)
        prob = torch.softmax(logits, dim=1)[:, 1]
        return {"cls": logits, "prob": prob, "feat": feats}

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
