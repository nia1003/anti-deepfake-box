"""
Evaluation metrics: AUC, ACC, EER, AP.
Compatible with DeepfakeBench's metric_scoring convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class DetectionMetrics:
    """Complete metrics bundle for one evaluation run."""
    auc: float
    acc: float
    eer: float
    ap: float
    threshold: float        # Threshold used for ACC / EER
    n_real: int
    n_fake: int
    dataset: str = ""
    detector: str = ""
    modalities: str = ""

    def __str__(self) -> str:
        return (
            f"[{self.detector or 'ADB'} | {self.dataset}]\n"
            f"  AUC={self.auc:.4f}  ACC={self.acc:.4f}  "
            f"EER={self.eer:.4f}  AP={self.ap:.4f}\n"
            f"  n_real={self.n_real}  n_fake={self.n_fake}  "
            f"threshold={self.threshold:.3f}"
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "auc": self.auc, "acc": self.acc, "eer": self.eer, "ap": self.ap,
            "n_real": self.n_real, "n_fake": self.n_fake,
        }


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate and the corresponding threshold.

    Returns (eer, threshold).
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    threshold = float(thresholds[eer_idx]) if eer_idx < len(thresholds) else 0.5
    return eer, threshold


def compute_metrics(
    labels: List[int],
    scores: List[Optional[float]],
    threshold: float = 0.5,
    dataset: str = "",
    detector: str = "",
    modalities: str = "",
) -> DetectionMetrics:
    """
    Compute AUC, ACC, EER, AP from prediction scores.

    Parameters
    ----------
    labels  : list of 0/1 ground-truth labels
    scores  : list of fake probabilities (None entries are filtered out)
    threshold : decision threshold for ACC
    """
    # Filter out None predictions
    valid = [(l, s) for l, s in zip(labels, scores) if s is not None]
    if not valid:
        return DetectionMetrics(
            auc=0.0, acc=0.0, eer=1.0, ap=0.0,
            threshold=threshold, n_real=0, n_fake=0,
            dataset=dataset, detector=detector,
        )

    y_true = np.array([l for l, _ in valid])
    y_score = np.array([s for _, s in valid])
    y_pred = (y_score >= threshold).astype(int)

    # Guard against single-class inputs
    if len(np.unique(y_true)) < 2:
        return DetectionMetrics(
            auc=float(np.mean(y_pred == y_true)),
            acc=float(np.mean(y_pred == y_true)),
            eer=0.0, ap=0.0,
            threshold=threshold,
            n_real=int((y_true == 0).sum()),
            n_fake=int((y_true == 1).sum()),
            dataset=dataset, detector=detector,
        )

    auc = float(roc_auc_score(y_true, y_score))
    acc = float(accuracy_score(y_true, y_pred))
    eer, eer_thr = compute_eer(y_true, y_score)
    ap = float(average_precision_score(y_true, y_score))

    return DetectionMetrics(
        auc=auc, acc=acc, eer=eer, ap=ap,
        threshold=eer_thr,
        n_real=int((y_true == 0).sum()),
        n_fake=int((y_true == 1).sum()),
        dataset=dataset,
        detector=detector,
        modalities=modalities,
    )


def video_level_auc(
    video_scores: Dict[str, List[float]],
    video_labels: Dict[str, int],
) -> float:
    """
    Aggregate frame-level scores to video-level AUC via mean pooling.

    Parameters
    ----------
    video_scores : dict mapping video_name → list of per-frame fake scores
    video_labels : dict mapping video_name → 0/1 label
    """
    labels, scores = [], []
    for name, frame_scores in video_scores.items():
        if name in video_labels and frame_scores:
            labels.append(video_labels[name])
            scores.append(float(np.mean(frame_scores)))

    if not labels or len(np.unique(labels)) < 2:
        return 0.0
    return float(roc_auc_score(np.array(labels), np.array(scores)))
