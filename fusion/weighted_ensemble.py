"""
Weighted Ensemble Fusion.

Combines three detector scores into a single fake probability.
Gracefully handles None scores (unavailable modalities, e.g., no audio)
by excluding them and re-normalising remaining weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


DEFAULT_WEIGHTS = {
    "visual": 0.50,
    "rppg": 0.25,
    "sync": 0.25,
}


@dataclass
class FusionResult:
    """Structured output from the fusion module."""
    fake_score: float             # Final ensemble score ∈ [0, 1]
    is_fake: bool                 # Threshold decision
    threshold: float              # Decision threshold used
    scores: Dict[str, Optional[float]] = field(default_factory=dict)
    weights_used: Dict[str, float] = field(default_factory=dict)
    modalities_used: int = 0

    def __str__(self) -> str:
        lines = [
            f"Prediction : {'FAKE' if self.is_fake else 'REAL'}",
            f"Fake Score : {self.fake_score:.4f} (threshold={self.threshold:.2f})",
        ]
        for k, v in self.scores.items():
            w = self.weights_used.get(k, 0.0)
            val_str = f"{v:.4f}" if v is not None else "N/A (skipped)"
            lines.append(f"  {k:12s}: score={val_str}  weight={w:.3f}")
        return "\n".join(lines)


class WeightedEnsemble:
    """
    Compute weighted average of available detector scores.

    Config keys
    -----------
    fusion.weights.visual  : float (default 0.50)
    fusion.weights.rppg    : float (default 0.25)
    fusion.weights.sync    : float (default 0.25)
    fusion.threshold       : float (default 0.50)
    """

    def __init__(self, config: dict):
        fusion_cfg = config.get("fusion", {})
        weights_cfg = fusion_cfg.get("weights", {})
        self.weights: Dict[str, float] = {
            "visual": float(weights_cfg.get("visual", DEFAULT_WEIGHTS["visual"])),
            "rppg":   float(weights_cfg.get("rppg",   DEFAULT_WEIGHTS["rppg"])),
            "sync":   float(weights_cfg.get("sync",   DEFAULT_WEIGHTS["sync"])),
        }
        self.threshold: float = float(fusion_cfg.get("threshold", 0.50))

    def fuse(
        self,
        visual_score: Optional[float] = None,
        rppg_score: Optional[float] = None,
        sync_score: Optional[float] = None,
    ) -> FusionResult:
        """
        Fuse up to three detector scores.

        Any None score is excluded; remaining weights are re-normalised.
        If all scores are None, returns fake_score=0.5 (uncertain).
        """
        raw_scores: Dict[str, Optional[float]] = {
            "visual": visual_score,
            "rppg": rppg_score,
            "sync": sync_score,
        }

        # Filter available modalities
        active = {k: v for k, v in raw_scores.items() if v is not None}

        if not active:
            return FusionResult(
                fake_score=0.5,
                is_fake=False,
                threshold=self.threshold,
                scores=raw_scores,
                weights_used={k: 0.0 for k in raw_scores},
                modalities_used=0,
            )

        total_weight = sum(self.weights[k] for k in active)
        if total_weight == 0:
            total_weight = 1.0  # Uniform fallback

        weights_used: Dict[str, float] = {}
        ensemble_score = 0.0
        for k, score in active.items():
            w = self.weights[k] / total_weight
            weights_used[k] = w
            ensemble_score += w * score

        for k in raw_scores:
            if k not in weights_used:
                weights_used[k] = 0.0

        return FusionResult(
            fake_score=float(ensemble_score),
            is_fake=ensemble_score >= self.threshold,
            threshold=self.threshold,
            scores=raw_scores,
            weights_used=weights_used,
            modalities_used=len(active),
        )

    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """Update fusion weights (e.g., after calibration on validation set)."""
        for k, v in new_weights.items():
            if k in self.weights:
                self.weights[k] = float(v)
