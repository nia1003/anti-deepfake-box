"""
Audio-Visual Synchronisation Detector based on LatentSync SyncNet.

Principle: deepfake lip-sync videos often have temporal desynchronisation
between mouth movements and the audio track. LatentSync's SyncNet was trained
to measure audio-visual correspondence; low sync confidence → high fake score.

Pipeline:
1. Check audio availability (graceful degradation: return None if no audio)
2. Extract audio → 16kHz WAV
3. Whisper tiny → mel spectrogram → audio embeddings
4. InsightFace crops (256×256) sampled at 25fps (already in FaceTrack)
5. SyncNet forward → sync_confidence ∈ [0, 1]
6. fake_score = 1 − sync_confidence
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .base_detector import BaseDetector
from preprocessing.face_extractor import FaceTrack

_LATENTSYNC_PATH = Path(__file__).parent.parent / "third_party" / "latentsync"
if _LATENTSYNC_PATH.exists():
    sys.path.insert(0, str(_LATENTSYNC_PATH))


class SyncDetector(BaseDetector):
    """
    Audio-visual synchronisation detector using LatentSync SyncNet.

    Returns None when no audio track is available (caller handles
    graceful degradation in the fusion module).
    """

    WINDOW_FRAMES: int = 25        # ~1 second at 25fps per SyncNet window
    HOP_FRAMES: int = 5            # stride between windows
    AUDIO_SR: int = 16000          # Whisper expects 16kHz

    def __init__(self, config: dict):
        super().__init__(config)
        self.syncnet_path: str = config.get("syncnet_path", "")
        self.whisper_model: str = config.get("whisper_model", "tiny")
        self.whisper_device: str = config.get("whisper_device", "cpu")  # CPU to avoid GPU clash
        self.model = None
        self._whisper = None

    def load(self) -> None:
        # Load SyncNet
        try:
            from latentsync.models.stable_syncnet import StableSyncNet
            self.model = StableSyncNet()
            if self.syncnet_path and Path(self.syncnet_path).exists():
                state = torch.load(self.syncnet_path, map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                self.model.load_state_dict(state, strict=False)
            self.model.eval()
            self.model = self.model.to(self.device)
        except ImportError:
            self.model = None

        # Load Whisper (on CPU by default to avoid VRAM contention)
        try:
            import whisper
            self._whisper = whisper.load_model(
                self.whisper_model, device=self.whisper_device
            )
        except ImportError:
            self._whisper = None

    def _extract_audio_features(self, audio_path: str) -> Optional[np.ndarray]:
        """Whisper → mel → audio feature array (T_audio, D)."""
        if self._whisper is None:
            return None
        try:
            import whisper
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio)  # (80, 3000)
            return mel.numpy()
        except Exception:
            return None

    def _sliding_window_scores(
        self,
        crops_256: np.ndarray,     # (T, 256, 256, 3)
        audio_features: np.ndarray,  # (80, T_mel)
    ) -> float:
        """
        Run SyncNet over sliding windows; return mean sync confidence.
        If model unavailable, fall back to a heuristic based on facial motion.
        """
        if self.model is None:
            return self._motion_heuristic(crops_256)

        T = len(crops_256)
        scores = []

        # Segment mel into per-window chunks aligned with video frames
        mel_per_frame = audio_features.shape[1] / max(T, 1)

        for start in range(0, T - self.WINDOW_FRAMES + 1, self.HOP_FRAMES):
            end = start + self.WINDOW_FRAMES
            video_chunk = crops_256[start:end]  # (W, 256, 256, 3)

            mel_start = int(start * mel_per_frame)
            mel_end = int(end * mel_per_frame)
            audio_chunk = audio_features[:, mel_start:mel_end]

            # Normalise video: (1, 3, W, 256, 256) float32 [-1,1]
            v = torch.from_numpy(video_chunk.astype(np.float32) / 127.5 - 1.0)
            v = v.permute(3, 0, 1, 2).unsqueeze(0).to(self.device)

            # Audio: (1, 80, W_mel)
            # Resize mel to match window duration if needed
            a_len = audio_chunk.shape[1]
            if a_len == 0:
                continue
            a = torch.from_numpy(audio_chunk).float().unsqueeze(0).to(self.device)

            try:
                with torch.no_grad():
                    out = self.model(v, a)
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    score = torch.sigmoid(out).mean().item()
                scores.append(score)
            except Exception:
                continue

        return float(np.mean(scores)) if scores else 0.5

    @staticmethod
    def _motion_heuristic(crops_256: np.ndarray) -> float:
        """
        Fallback: compute temporal variance of mouth region.
        Low variance → static lip region → potential sync issue.
        Returns a pseudo sync score in [0, 1].
        """
        T = len(crops_256)
        if T < 2:
            return 0.5
        # Crop mouth region (approx. bottom-third of face)
        mouth = crops_256[:, 170:240, 80:176, :]  # (T, H, W, 3)
        gray = mouth.mean(axis=-1)  # (T, H, W)
        diffs = np.abs(np.diff(gray.reshape(T, -1), axis=0)).mean()
        # Normalise: typical real video has diffs ~5-15, deepfakes ~1-5
        score = float(np.clip(diffs / 10.0, 0.0, 1.0))
        return score

    def _detect_impl(
        self,
        face_track: FaceTrack,
        audio_path: Optional[str] = None,
    ) -> Optional[float]:
        if audio_path is None:
            return None  # No audio → modality unavailable

        if face_track is None or face_track.T == 0:
            return None

        audio_feats = self._extract_audio_features(audio_path)
        if audio_feats is None:
            return None

        sync_confidence = self._sliding_window_scores(face_track.crops_256, audio_feats)
        fake_score = 1.0 - sync_confidence
        return float(np.clip(fake_score, 0.0, 1.0))
