"""
TS-CAN rPPG detector stub for Anti-Deepfake-Box.

TS-CAN (Temporal Shift Convolutional Attention Network) from rPPG-Toolbox:
  Liu et al. "Multi-Task Temporal Shift Attention Networks for On-Device
  Contactless Vitals Measurement." NeurIPS 2020.

Status: rppg_toolbox — weights confirmation pending.

To activate:
  1. Clone rPPG-Toolbox into third_party/rppg_toolbox/
  2. Download pretrained TS-CAN checkpoint
  3. Implement _detect_impl() using rPPG-Toolbox's inference pipeline
  4. Update status in registry.py to "available"
"""

from __future__ import annotations

from typing import Optional

from .base_detector import BaseDetector
from preprocessing.face_extractor import FaceTrack


class TSCANDetector(BaseDetector):
    """
    TS-CAN-based rPPG deepfake detector.

    SNR pipeline (same as POS):
        rPPG signal extraction → Welch PSD → SNR → snr_to_fake_score()

    config keys:
        tscan_checkpoint : path to TS-CAN weights
        snr_threshold    : float, default 1.5 (calibrate via calibrate_snr.py)
        snr_scale        : float, default 1.0
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.checkpoint: str = config.get("tscan_checkpoint", "")
        self.snr_threshold: float = config.get("snr_threshold", 1.5)
        self.snr_scale: float = config.get("snr_scale", 1.0)

    def load(self) -> None:
        raise NotImplementedError(
            "TS-CAN detector is not yet implemented.\n"
            "Steps to activate:\n"
            "  1. Clone rPPG-Toolbox: git clone https://github.com/ubicomplab/rPPG-Toolbox "
            "third_party/rppg_toolbox\n"
            "  2. Download TS-CAN checkpoint and set 'tscan_checkpoint' in config\n"
            "  3. Implement TSCANDetector._detect_impl() in detectors/rppg_tscan_detector.py\n"
            "  4. Change status to 'available' in detectors/registry.py"
        )

    def _detect_impl(
        self,
        face_track: FaceTrack,
        audio_path: Optional[str] = None,
    ) -> Optional[float]:
        raise NotImplementedError("TS-CAN not yet implemented; call load() first.")
