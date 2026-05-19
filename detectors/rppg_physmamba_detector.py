"""
PhysMamba rPPG detector stub for Anti-Deepfake-Box.

PhysMamba is a Mamba-based rPPG model approved by the GP validator as an
alternative to POS. Implementation pending.

Status: planned

To activate:
  1. Obtain PhysMamba implementation and pretrained checkpoint
  2. Implement _detect_impl() with the same SNR pipeline as RPPGDetector (POS)
  3. Update status in registry.py to "available" or "rppg_toolbox"
"""

from __future__ import annotations

from typing import Optional

from .base_detector import BaseDetector
from preprocessing.face_extractor import FaceTrack


class PhysMambaDetector(BaseDetector):
    """
    PhysMamba-based rPPG deepfake detector (stub).

    config keys:
        physmamba_checkpoint : path to PhysMamba weights
        snr_threshold        : float, default 1.5
        snr_scale            : float, default 1.0
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.checkpoint: str = config.get("physmamba_checkpoint", "")
        self.snr_threshold: float = config.get("snr_threshold", 1.5)
        self.snr_scale: float = config.get("snr_scale", 1.0)

    def load(self) -> None:
        raise NotImplementedError(
            "PhysMamba detector is not yet implemented.\n"
            "Obtain the PhysMamba model, implement _detect_impl(), and set "
            "status='available' in detectors/registry.py."
        )

    def _detect_impl(
        self,
        face_track: FaceTrack,
        audio_path: Optional[str] = None,
    ) -> Optional[float]:
        raise NotImplementedError("PhysMamba not yet implemented; call load() first.")
