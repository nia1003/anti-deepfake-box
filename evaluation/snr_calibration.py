"""
SNR Threshold Calibration for the rPPG Detector.

Uses FF++ validation split to find the optimal SNR threshold that maximises
Youden's J statistic (= TPR - FPR), then updates configs/default.yaml.

Usage:
    python scripts/calibrate_snr.py \
        --data_root /data/FF++ \
        --config configs/default.yaml \
        --split val
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.metrics import roc_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def youden_threshold(
    labels: np.ndarray,
    snr_values: np.ndarray,
) -> Tuple[float, float]:
    """
    Find SNR threshold that maximises Youden's J = TPR - FPR.

    Note: lower SNR = more likely fake, so we invert the scoring direction
    (fake_score = −SNR for ROC purposes).

    Returns (optimal_threshold, j_statistic).
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for calibration")

    # Invert: lower SNR → higher fake score
    fake_scores = -snr_values
    fpr, tpr, thresholds = roc_curve(labels, fake_scores, pos_label=1)

    j = tpr - fpr
    best_idx = int(np.argmax(j))
    optimal_fake_score_threshold = float(thresholds[best_idx])
    optimal_snr_threshold = -optimal_fake_score_threshold  # back to SNR space
    j_stat = float(j[best_idx])

    return optimal_snr_threshold, j_stat


def calibrate_snr_threshold(
    snr_values: List[float],
    labels: List[int],
) -> Dict:
    """
    Calibrate SNR threshold from collected (snr, label) pairs.

    Parameters
    ----------
    snr_values : list of SNR values (float, one per video)
    labels     : list of 0/1 ground-truth labels (0=real, 1=fake)

    Returns
    -------
    dict with keys: threshold, j_statistic, n_real, n_fake
    """
    arr_snr = np.array(snr_values, dtype=np.float64)
    arr_lbl = np.array(labels, dtype=np.int32)

    threshold, j_stat = youden_threshold(arr_lbl, arr_snr)

    return {
        "snr_threshold": float(threshold),
        "j_statistic": float(j_stat),
        "n_real": int((arr_lbl == 0).sum()),
        "n_fake": int((arr_lbl == 1).sum()),
        "snr_mean_real": float(arr_snr[arr_lbl == 0].mean()),
        "snr_mean_fake": float(arr_snr[arr_lbl == 1].mean()),
    }


def update_config_threshold(config_path: str, threshold: float) -> None:
    """Patch snr_threshold in a YAML config file in-place."""
    try:
        import yaml
    except ImportError:
        print("PyYAML not installed. Update snr_threshold manually.")
        return

    path = Path(config_path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("detectors", {}).setdefault("rppg", {})["snr_threshold"] = float(threshold)

    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    print(f"Updated snr_threshold={threshold:.4f} in {config_path}")
