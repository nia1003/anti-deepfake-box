"""
FAR/FRR curve analysis per module variant.

Reads module_scores.json + module_labels.json, produces:
  eval/far_frr_curves/<module_id>.png
  updates eval/module_stats.json with EER + Max_FAR_FRR

CLI:
    python eval/far_frr_analysis.py [--scores eval/module_scores.json] [--labels eval/module_labels.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def compute_far_frr_curve(
    scores: Dict[str, float],
    labels: Dict[str, int],
    n_thresholds: int = 101,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute FAR and FRR across linearly-spaced thresholds [0, 1].

    fake_score direction: higher = more fake.
        FAR(T) = P(fake_score < T  | video is fake)   ← fake wrongly judged REAL
        FRR(T) = P(fake_score >= T | video is real)   ← real wrongly judged FAKE

    Returns (thresholds, FAR_array, FRR_array) — all length n_thresholds.
    """
    vid_ids = [v for v in scores if v in labels]
    s = np.array([scores[v] for v in vid_ids], dtype=np.float64)
    y = np.array([labels[v] for v in vid_ids], dtype=np.int32)

    fake_mask = y == 1
    real_mask = y == 0
    total_fake = int(fake_mask.sum())
    total_real = int(real_mask.sum())

    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    FAR = np.zeros(n_thresholds)
    FRR = np.zeros(n_thresholds)

    for k, t in enumerate(thresholds):
        FAR[k] = (s[fake_mask] < t).sum() / max(total_fake, 1)
        FRR[k] = (s[real_mask] >= t).sum() / max(total_real, 1)

    return thresholds, FAR, FRR


def compute_eer(
    thresholds: np.ndarray, FAR: np.ndarray, FRR: np.ndarray
) -> Tuple[float, float]:
    """Return (EER_value, EER_threshold)."""
    diff = np.abs(FAR - FRR)
    idx = int(np.argmin(diff))
    eer = float((FAR[idx] + FRR[idx]) / 2.0)
    return eer, float(thresholds[idx])


def compute_max_far_frr(FAR: np.ndarray, FRR: np.ndarray) -> Tuple[float, float]:
    """Return (Max_FAR×FRR, threshold index)."""
    product = FAR * FRR
    idx = int(np.argmax(product))
    return float(product[idx]), idx


def plot_curve(
    module_id: str,
    thresholds: np.ndarray,
    FAR: np.ndarray,
    FRR: np.ndarray,
    eer: float,
    eer_thresh: float,
    max_far_frr: float,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"FAR / FRR Analysis — {module_id}", fontsize=13)

    # Left: FAR + FRR vs threshold
    ax = axes[0]
    ax.plot(thresholds, FAR, label="FAR (fake→REAL miss)", color="tab:red")
    ax.plot(thresholds, FRR, label="FRR (real→FAKE miss)", color="tab:blue")
    ax.axvline(eer_thresh, color="gray", linestyle="--", alpha=0.7, label=f"EER thresh={eer_thresh:.2f}")
    ax.scatter([eer_thresh], [eer], color="black", zorder=5)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Rate")
    ax.set_title(f"EER = {eer:.4f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: FAR × FRR product (con_b)
    ax2 = axes[1]
    product = FAR * FRR
    ax2.plot(thresholds, product, color="tab:purple")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("FAR × FRR")
    ax2.set_title(f"Max(FAR×FRR) = {max_far_frr:.6f}")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=100)
    plt.close(fig)


def run_analysis(
    scores_path: Path,
    labels_path: Path,
    output_dir: Path,
    stats_path: Path,
) -> None:
    with open(scores_path) as f:
        all_scores: Dict[str, Dict[str, float]] = json.load(f)
    with open(labels_path) as f:
        labels: Dict[str, int] = json.load(f)

    existing_stats: Dict[str, dict] = {}
    if stats_path.exists():
        with open(stats_path) as f:
            existing_stats = json.load(f)

    curves_dir = output_dir / "far_frr_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    updated_stats = dict(existing_stats)

    for module_id, scores in all_scores.items():
        print(f"[far_frr] {module_id} ...", end=" ", flush=True)
        thresholds, FAR, FRR = compute_far_frr_curve(scores, labels)
        eer, eer_thresh = compute_eer(thresholds, FAR, FRR)
        max_far_frr, _ = compute_max_far_frr(FAR, FRR)

        plot_curve(
            module_id=module_id,
            thresholds=thresholds,
            FAR=FAR,
            FRR=FRR,
            eer=eer,
            eer_thresh=eer_thresh,
            max_far_frr=max_far_frr,
            out_path=curves_dir / f"{module_id}.png",
        )

        entry = dict(updated_stats.get(module_id, {}))
        entry["EER"] = eer
        entry["EER_threshold"] = eer_thresh
        entry["Max_FAR_FRR"] = max_far_frr
        updated_stats[module_id] = entry

        print(f"EER={eer:.4f}  Max_FAR×FRR={max_far_frr:.6f}  → {curves_dir/module_id}.png")

    with open(stats_path, "w") as f:
        json.dump(updated_stats, f, indent=2)
    print(f"\n[far_frr] Stats updated → {stats_path}")


def main():
    ap = argparse.ArgumentParser(description="Compute FAR/FRR curves for each module variant")
    ap.add_argument("--scores",  type=Path, default=Path("eval/module_scores.json"))
    ap.add_argument("--labels",  type=Path, default=Path("eval/module_labels.json"))
    ap.add_argument("--outdir",  type=Path, default=Path("eval"))
    ap.add_argument("--stats",   type=Path, default=Path("eval/module_stats.json"))
    args = ap.parse_args()

    run_analysis(
        scores_path=args.scores,
        labels_path=args.labels,
        output_dir=args.outdir,
        stats_path=args.stats,
    )


if __name__ == "__main__":
    main()
