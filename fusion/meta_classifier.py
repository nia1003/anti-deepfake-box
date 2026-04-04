"""
Meta-Classifier Fusion (learned fusion).

A 2-layer MLP trained on top of three detector outputs to learn inter-modality
correlations that weighted averaging cannot capture.

Activated only after Stage 1 validation passes (AUC > 0.95 on FF++ and
AUC > 0.85 on Celeb-DF v2). Use WeightedEnsemble for initial validation.

Input modes (config: fusion.meta.input_mode):
  'scores'   : 3-dim vector [visual, rppg, sync] (missing → 0)
  'features' : concatenated feature vectors from each detector backbone
  'both'     : scores + features
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .weighted_ensemble import FusionResult


class MetaClassifierNet(nn.Module):
    """2-layer MLP for learned score fusion."""

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MetaClassifier:
    """
    Learned meta-classifier fusion module.

    Config keys
    -----------
    fusion.meta.input_mode  : 'scores' | 'features' | 'both'
    fusion.meta.hidden_dims : list of ints, e.g. [512, 256]
    fusion.meta.dropout     : float, default 0.3
    fusion.threshold        : float, default 0.50
    fusion.meta.model_path  : path to saved weights (optional)
    """

    def __init__(self, config: dict):
        fusion_cfg = config.get("fusion", {})
        meta_cfg = fusion_cfg.get("meta", {})

        self.input_mode: str = meta_cfg.get("input_mode", "scores")
        self.hidden_dims: List[int] = meta_cfg.get("hidden_dims", [512, 256])
        self.dropout: float = float(meta_cfg.get("dropout", 0.3))
        self.threshold: float = float(fusion_cfg.get("threshold", 0.50))
        self.device: str = config.get("device", "cuda")
        self.model_path: str = meta_cfg.get("model_path", "")

        self._input_dim = self._resolve_input_dim()
        self.model: Optional[MetaClassifierNet] = None

    def _resolve_input_dim(self) -> int:
        if self.input_mode == "scores":
            return 3  # visual, rppg, sync
        elif self.input_mode == "features":
            # Defaults: xception=2048, physnet=32, syncnet=512 → 2592
            return 2592
        else:  # 'both'
            return 3 + 2592

    def build(self, input_dim: Optional[int] = None) -> None:
        """Instantiate the MLP (must be called before train or load)."""
        dim = input_dim or self._input_dim
        self.model = MetaClassifierNet(dim, self.hidden_dims, self.dropout)
        self.model.to(self.device)

    def load(self, path: Optional[str] = None) -> None:
        """Load trained weights from disk."""
        p = path or self.model_path
        if not p:
            raise ValueError("No model path specified for MetaClassifier.load()")
        if self.model is None:
            self.build()
        state = torch.load(p, map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        self.model.eval()

    def save(self, path: str) -> None:
        """Save model weights."""
        if self.model is None:
            raise RuntimeError("Model not built yet.")
        torch.save({"model_state_dict": self.model.state_dict()}, path)

    def _scores_to_tensor(
        self,
        visual: Optional[float],
        rppg: Optional[float],
        sync: Optional[float],
    ) -> torch.Tensor:
        vals = [
            visual if visual is not None else 0.5,
            rppg   if rppg   is not None else 0.5,
            sync   if sync   is not None else 0.5,
        ]
        return torch.tensor(vals, dtype=torch.float32)

    def fuse(
        self,
        visual_score: Optional[float] = None,
        rppg_score: Optional[float] = None,
        sync_score: Optional[float] = None,
        feature_vector: Optional[np.ndarray] = None,
    ) -> FusionResult:
        """
        Fuse scores (and optionally feature vectors) via the trained MLP.

        feature_vector : concatenated backbone features (numpy 1-D array)
        """
        if self.model is None:
            raise RuntimeError(
                "MetaClassifier not loaded. Call build() + load(), or train first."
            )

        if self.input_mode == "scores":
            x = self._scores_to_tensor(visual_score, rppg_score, sync_score)
        elif self.input_mode == "features":
            if feature_vector is None:
                raise ValueError("feature_vector required for input_mode='features'")
            x = torch.from_numpy(feature_vector.astype(np.float32))
        else:  # 'both'
            s = self._scores_to_tensor(visual_score, rppg_score, sync_score)
            f = torch.from_numpy(feature_vector.astype(np.float32)) if feature_vector is not None \
                else torch.zeros(self._input_dim - 3)
            x = torch.cat([s, f])

        x = x.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            prob = F.softmax(logits, dim=1)[0, 1].item()

        return FusionResult(
            fake_score=prob,
            is_fake=prob >= self.threshold,
            threshold=self.threshold,
            scores={"visual": visual_score, "rppg": rppg_score, "sync": sync_score},
            weights_used={},
            modalities_used=sum(v is not None for v in [visual_score, rppg_score, sync_score]),
        )

    # ------------------------------------------------------------------ #
    #  Training helpers (called by scripts/train_fusion.py)               #
    # ------------------------------------------------------------------ #

    def train_epoch(
        self,
        data_loader,
        optimizer: torch.optim.Optimizer,
        criterion=None,
    ) -> float:
        """Run one training epoch. Returns mean loss."""
        if self.model is None:
            raise RuntimeError("Call build() first.")
        self.model.train()
        criterion = criterion or nn.CrossEntropyLoss()
        total_loss = 0.0

        for batch in data_loader:
            x, labels = batch
            x = x.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            logits = self.model(x)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / max(len(data_loader), 1)
