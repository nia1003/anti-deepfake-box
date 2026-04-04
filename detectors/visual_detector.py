"""
Visual Deepfake Detector based on FaceForensics++ XceptionNet.

Wraps the XceptionNet classifier from third_party/faceforensics.
Accepts FaceTrack (SSOT crops_299) and returns per-video fake score.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from .base_detector import BaseDetector
from preprocessing.face_extractor import FaceTrack

# Add FaceForensics to path when available
_FF_PATH = Path(__file__).parent.parent / "third_party" / "faceforensics" / "classification"
if _FF_PATH.exists():
    sys.path.insert(0, str(_FF_PATH))


class VisualDetector(BaseDetector):
    """
    XceptionNet-based visual artifact detector.

    Trained on FaceForensics++ c23 (Deepfakes, Face2Face, FaceSwap, NeuralTextures).
    Inputs: face crops at 299×299; output: softmax fake probability.

    Detection logic:
    1. Take FaceTrack.crops_299 (T, 299, 299, 3)
    2. Sample up to max_frames evenly-spaced frames
    3. Normalise (ImageNet mean/std)
    4. Batch forward through XceptionNet
    5. Softmax → class 1 (fake) probability per frame
    6. Return weighted-mean fake score
    """

    MEAN = [0.5, 0.5, 0.5]
    STD = [0.5, 0.5, 0.5]

    def __init__(self, config: dict):
        super().__init__(config)
        self.pretrained_path: str = config.get("visual_pretrained", "")
        self.max_frames: int = config.get("visual_max_frames", 32)
        self.batch_size: int = config.get("visual_batch_size", 16)
        self.model = None

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

    def load(self) -> None:
        try:
            from network.xception import xception
            model = xception(num_classes=2, pretrained=False)
        except ImportError:
            model = self._build_fallback_model()

        if self.pretrained_path and Path(self.pretrained_path).exists():
            state = torch.load(self.pretrained_path, map_location="cpu")
            # Handle various checkpoint formats
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            elif isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            # Handle pointwise conv weight shape mismatch (FF++ convention)
            for k, v in list(state.items()):
                if "pointwise" in k and v.dim() == 2:
                    state[k] = v.unsqueeze(-1).unsqueeze(-1)
            model.load_state_dict(state, strict=False)

        model.eval()
        self.model = model.to(self.device)

    def _build_fallback_model(self):
        """Minimal EfficientNet-B0 fallback when XceptionNet unavailable."""
        try:
            import torchvision.models as models
            m = models.efficientnet_b0(pretrained=False)
            m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, 2)
            return m
        except Exception:
            raise ImportError(
                "Cannot load XceptionNet. Add third_party/faceforensics to PYTHONPATH "
                "or install torchvision."
            )

    def _preprocess(self, crops_299: np.ndarray) -> torch.Tensor:
        """(T, 299, 299, 3) uint8 → (T, 3, 299, 299) float32 tensor."""
        from PIL import Image as PILImage
        tensors = []
        for crop in crops_299:
            img = PILImage.fromarray(crop)
            tensors.append(self.transform(img))
        return torch.stack(tensors)

    def _sample_frames(self, crops: np.ndarray) -> np.ndarray:
        """Evenly sample max_frames from the track."""
        T = len(crops)
        if T <= self.max_frames:
            return crops
        indices = np.linspace(0, T - 1, self.max_frames, dtype=int)
        return crops[indices]

    def _detect_impl(
        self,
        face_track: FaceTrack,
        audio_path: Optional[str] = None,
    ) -> Optional[float]:
        if face_track is None or face_track.T == 0:
            return None

        sampled = self._sample_frames(face_track.crops_299)
        tensor = self._preprocess(sampled)  # (T, 3, 299, 299)

        fake_probs = []
        with torch.no_grad():
            for start in range(0, len(tensor), self.batch_size):
                batch = tensor[start: start + self.batch_size].to(self.device)
                logits = self.model(batch)
                prob = F.softmax(logits, dim=1)[:, 1]  # fake class
                fake_probs.extend(prob.cpu().numpy().tolist())

        return float(np.mean(fake_probs))
