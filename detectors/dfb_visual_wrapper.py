"""
Generic wrapper for DeepfakeBench visual detectors.

Allows ADB's collect_scores.py / inference.py to run any DFB visual detector
(UCF, SBI, F3Net, SPSL, SRM, EfficientB4, FaceXray, LSDA) using ADB's
SSOT face extraction pipeline.

Requirements
------------
1. DeepfakeBench repo accessible at $DFB_PATH/training, or as a sibling
   directory named DeepfakeBench/ or deepfakebench/.
   Set the env var:  export DFB_PATH=/home/user/DeepfakeBench
2. DFB pretrained checkpoint passed via config["dfb_pretrained"].
   (config["pretrained"] is also checked as a fallback.)
3. DFB detector config YAML at $DFB_PATH/training/config/detector/{name}.yaml.

Input resolution
----------------
Uses FaceTrack.aligned_256 (256×256 InsightFace-aligned crops, uint8 RGB).
Normalised to [-1, 1] matching DFB's default DatasetTransform.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .base_detector import BaseDetector
from preprocessing.face_extractor import FaceTrack

_ADB_ROOT = Path(__file__).parent.parent


def _find_dfb_training_path() -> Optional[str]:
    """Return path to DFB's training/ directory, or None if not found."""
    # 1. Env var DFB_PATH=/path/to/DeepfakeBench
    env = os.environ.get("DFB_PATH", "")
    if env:
        p = Path(env) / "training"
        if p.is_dir():
            return str(p)
    # 2. Sibling directory
    for name in ("DeepfakeBench", "deepfakebench"):
        p = _ADB_ROOT.parent / name / "training"
        if p.is_dir():
            return str(p)
    return None


@contextlib.contextmanager
def _dfb_path_ctx(dfb_training: str):
    """
    Temporarily insert DFB's training dir at the front of sys.path,
    stashing ADB's 'detectors' package to avoid the package-name collision.

    Within this context:
      - `import detectors` resolves to DFB's detectors (giving access to DETECTOR registry)
      - `from metrics.registry import DETECTOR` resolves to DFB's metrics

    After the context exits:
      - ADB's detectors modules are restored in sys.modules
      - DFB's detectors modules are removed from sys.modules
      - The instantiated DFB detector object retains its class (safe: class globals
        were bound at import time and don't re-query sys.modules at forward()-time)
    """
    adb_mods = {k: v for k, v in list(sys.modules.items())
                if k == "detectors" or k.startswith("detectors.")}
    for k in list(adb_mods):
        del sys.modules[k]

    sys.path.insert(0, dfb_training)
    try:
        yield
    finally:
        # Remove DFB-loaded detectors entries
        for k in [k for k in list(sys.modules) if k == "detectors" or k.startswith("detectors.")]:
            del sys.modules[k]
        try:
            sys.path.remove(dfb_training)
        except ValueError:
            pass
        # Restore ADB's detectors
        sys.modules.update(adb_mods)


class DFBVisualDetector(BaseDetector):
    """
    Wraps any DeepfakeBench visual detector for use in ADB's pipeline.

    Supported detector names (must match DFB YAML model_name and registry key):
        xception, ucf, sbi, f3net, spsl, srm, efficientnetb4, facexray, lsda

    config keys (passed through from collect_scores / inference):
        device          : "cuda" | "cpu"
        dfb_pretrained  : path to DFB checkpoint (overrides YAML's pretrained field)
        pretrained      : alias for dfb_pretrained
    """

    def __init__(self, dfb_detector_name: str, config: dict):
        super().__init__(config)
        self.dfb_detector_name = dfb_detector_name
        self._dfb_det: Optional[torch.nn.Module] = None
        self._dfb_training = _find_dfb_training_path()

    def load(self) -> None:
        if self._dfb_training is None:
            raise ImportError(
                f"DeepfakeBench not found. To use '{self.dfb_detector_name}':\n"
                "  export DFB_PATH=/path/to/DeepfakeBench\n"
                "or place DeepfakeBench/ as a sibling of anti-deepfake-box/."
            )

        cfg_path = (
            Path(self._dfb_training) / "config" / "detector"
            / f"{self.dfb_detector_name}.yaml"
        )
        if not cfg_path.exists():
            available = sorted(p.stem for p in (Path(self._dfb_training) / "config" / "detector").glob("*.yaml"))
            raise FileNotFoundError(
                f"DFB detector config not found: {cfg_path}\n"
                f"Available DFB detectors: {available}"
            )

        with _dfb_path_ctx(self._dfb_training):
            import yaml
            with open(cfg_path) as f:
                dfb_cfg = yaml.safe_load(f)

            # Let ADB config override pretrained path
            pretrained = self.config.get("dfb_pretrained", self.config.get("pretrained", ""))
            if pretrained:
                dfb_cfg["pretrained"] = str(pretrained)
            dfb_cfg.setdefault("device", self.device)

            from detectors import DETECTOR as _DFB_DET
            cls = _DFB_DET[self.dfb_detector_name]
            self._dfb_det = cls(dfb_cfg)

        self._dfb_det.eval()
        try:
            self._dfb_det = self._dfb_det.to(self.device)
        except Exception:
            pass  # Some DFB detectors don't subclass nn.Module uniformly

    def _to_dfb_batch(self, face_track: FaceTrack) -> dict:
        """
        Convert FaceTrack.aligned_256 → DFB-style data_dict.

        DFB's DatasetTransform normalises to [-1, 1] via (x/255 - 0.5) / 0.5.
        We replicate that here so DFB detectors see identical input statistics.
        """
        crops = face_track.aligned_256  # (T, 256, 256, 3) uint8
        x = torch.from_numpy(crops.astype(np.float32) / 127.5 - 1.0)
        x = x.permute(0, 3, 1, 2)  # (T, 3, 256, 256)
        return {
            "image":    x.to(self.device),
            "label":    torch.zeros(len(crops), dtype=torch.long, device=self.device),
            "landmark": None,
            "mask":     None,
        }

    def _detect_impl(
        self,
        face_track: FaceTrack,
        audio_path: Optional[str] = None,
    ) -> Optional[float]:
        if face_track is None or face_track.T == 0:
            return None
        if not hasattr(face_track, "aligned_256") or face_track.aligned_256 is None:
            return None
        if len(face_track.aligned_256) == 0:
            return None

        data_dict = self._to_dfb_batch(face_track)
        with torch.no_grad():
            pred_dict = self._dfb_det.forward(data_dict, inference=True)

        prob = pred_dict.get("prob")
        if prob is None:
            # Some DFB detectors return cls logits only
            cls_logits = pred_dict.get("cls")
            if cls_logits is not None:
                prob = torch.softmax(cls_logits, dim=1)[:, 1]
            else:
                return None

        return float(prob.mean().item())
