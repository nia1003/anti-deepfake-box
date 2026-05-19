"""
4-Stage Weak Classifier Selection Pipeline.

Takes per-classifier score CSVs (produced by scripts/collect_scores.py) and
returns the set of classifiers that survive all selection stages:

  Stage 1 — AUC filter         drop if AUC < min_auc (default 0.55)
  Stage 2 — Correlation filter  drop redundant classifiers (|corr| > max_corr)
  Stage 3 — FAR/FRR analysis    inspect complementarity at key operating points
  Stage 4 — Pareto filter       drop dominated classifiers (worse AUC + worse EER)

The selected set is the input to the GP solver cascade design step.

Usage
-----
from fusion.cascade_selection import select_classifiers

retained = select_classifiers(
    csv_dir="results/",          # directory with visual/rppg/sync_scores.csv
    label_csv="labels.csv",      # ground-truth (sample_id, label)
    min_auc=0.55,
    max_corr=0.90,
    output_json="results/selected_classifiers.json",
)
# retained → e.g. ["visual", "sync"]
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve

import sys
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.metrics import compute_metrics, compute_eer, DetectionMetrics


# ------------------------------------------------------------------ #
#  CSV loading helpers                                                 #
# ------------------------------------------------------------------ #

def _load_scores_csv(path: str) -> Tuple[List[str], List[float], List[int]]:
    """
    Load a 12-column score CSV produced by collect_scores.py.

    Returns (sample_ids, scores, labels).
    Rows with status="failed" or empty fake_score are excluded from the
    returned arrays but their sample_ids are still tracked for alignment
    diagnostics.
    """
    sample_ids, scores, labels = [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status", "ok") != "ok":
                continue
            score_str = row.get("fake_score", "").strip()
            label_str = row.get("label", "").strip()
            if not score_str or not label_str:
                continue
            try:
                sample_ids.append(row["sample_id"].strip())
                scores.append(float(score_str))
                labels.append(int(label_str))
            except (ValueError, KeyError):
                continue
    return sample_ids, scores, labels


def _load_all_csvs(
    csv_dir: str,
    names: Optional[List[str]] = None,
) -> Dict[str, Tuple[List[str], np.ndarray, np.ndarray]]:
    """
    Load all modality CSVs from csv_dir.

    Returns {name: (sample_ids, scores_array, labels_array)}.
    """
    if names is None:
        names = ["visual", "rppg", "sync"]

    result: Dict[str, Tuple[List[str], np.ndarray, np.ndarray]] = {}
    for name in names:
        csv_path = Path(csv_dir) / f"{name}_scores.csv"
        if not csv_path.exists():
            print(f"  [cascade_selection] {csv_path} not found — skipping {name}")
            continue
        ids, scores, labels = _load_scores_csv(str(csv_path))
        if not ids:
            print(f"  [cascade_selection] {csv_path} has no valid rows — skipping {name}")
            continue
        result[name] = (ids, np.array(scores, dtype=np.float64), np.array(labels, dtype=np.int32))
        print(f"  [cascade_selection] loaded {name}: {len(ids)} samples "
              f"  (real={sum(l==0 for l in labels)}, fake={sum(l==1 for l in labels)})")
    return result


# ------------------------------------------------------------------ #
#  Stage 1: AUC filter                                                 #
# ------------------------------------------------------------------ #

def filter_by_auc(
    data: Dict[str, Tuple[List[str], np.ndarray, np.ndarray]],
    min_auc: float = 0.55,
) -> Tuple[Dict[str, Tuple[List[str], np.ndarray, np.ndarray]], Dict[str, DetectionMetrics]]:
    """
    Drop classifiers whose AUC < min_auc.

    Returns (filtered_data, metrics_per_classifier).
    """
    metrics: Dict[str, DetectionMetrics] = {}
    retained = {}
    for name, (ids, scores, labels) in data.items():
        m = compute_metrics(list(labels), list(scores), dataset="stage1_auc", detector=name)
        metrics[name] = m
        if m.auc >= min_auc:
            retained[name] = (ids, scores, labels)
            print(f"  Stage 1 ✓ {name}: AUC={m.auc:.4f} ≥ {min_auc}")
        else:
            print(f"  Stage 1 ✗ {name}: AUC={m.auc:.4f} < {min_auc}  → dropped")
    return retained, metrics


# ------------------------------------------------------------------ #
#  Stage 2: Correlation filter                                         #
# ------------------------------------------------------------------ #

def filter_by_correlation(
    data: Dict[str, Tuple[List[str], np.ndarray, np.ndarray]],
    metrics: Dict[str, DetectionMetrics],
    max_corr: float = 0.90,
) -> Dict[str, Tuple[List[str], np.ndarray, np.ndarray]]:
    """
    Greedily drop one member of any highly correlated pair.

    When |corr| > max_corr between classifier A and B, drop the one with
    lower AUC. Returns the filtered dict.
    """
    names = list(data.keys())
    scores_aligned = {}

    # Align all score arrays to a common sample_id ordering
    all_ids: List[str] = sorted({sid for ids, _, _ in data.values() for sid in ids})
    for name, (ids, scores, _) in data.items():
        id_to_score = dict(zip(ids, scores))
        # Use nan for missing samples (due to failed detections)
        scores_aligned[name] = np.array([id_to_score.get(sid, float("nan")) for sid in all_ids])

    # Build correlation matrix
    print(f"\n  Stage 2 correlation matrix:")
    dropped: set = set()
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            s1 = scores_aligned[n1]
            s2 = scores_aligned[n2]
            valid = ~(np.isnan(s1) | np.isnan(s2))
            if valid.sum() < 2:
                continue
            corr, _ = pearsonr(s1[valid], s2[valid])
            tag = " ← drop lower AUC" if abs(corr) > max_corr else ""
            print(f"    {n1} vs {n2}: r={corr:+.3f}{tag}")
            if abs(corr) > max_corr and n1 not in dropped and n2 not in dropped:
                auc1 = metrics.get(n1, None)
                auc2 = metrics.get(n2, None)
                if auc1 is not None and auc2 is not None:
                    to_drop = n1 if auc1.auc < auc2.auc else n2
                else:
                    to_drop = n2
                dropped.add(to_drop)
                print(f"    → dropped {to_drop} (lower AUC)")

    retained = {n: v for n, v in data.items() if n not in dropped}
    return retained


# ------------------------------------------------------------------ #
#  Stage 3: FAR/FRR complementarity analysis (informational)          #
# ------------------------------------------------------------------ #

def compute_far_frr_curves(
    data: Dict[str, Tuple[List[str], np.ndarray, np.ndarray]],
) -> Dict[str, Dict]:
    """
    Compute FAR/FRR operating curves per classifier.

    Returns dict: {name: {"thresholds": ..., "FAR": ..., "FRR": ..., "EER": ...}}
    """
    result = {}
    print("\n  Stage 3 FAR/FRR operating points:")
    for name, (_, scores, labels) in data.items():
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        far = fpr
        frr = 1.0 - tpr
        eer, eer_thr = compute_eer(labels, scores)
        result[name] = {
            "thresholds": thresholds.tolist(),
            "FAR": far.tolist(),
            "FRR": frr.tolist(),
            "EER": eer,
            "EER_threshold": eer_thr,
        }
        print(f"    {name}: EER={eer:.4f} @ thr={eer_thr:.3f}")
    return result


# ------------------------------------------------------------------ #
#  Stage 4: Pareto dominance filter                                    #
# ------------------------------------------------------------------ #

def pareto_filter(
    metrics: Dict[str, DetectionMetrics],
) -> List[str]:
    """
    Remove Pareto-dominated classifiers.

    A classifier X is dominated if there exists Y such that
    Y.auc >= X.auc AND Y.eer <= X.eer (both strictly not worse, at least one strict).

    Returns list of surviving classifier names.
    """
    names = list(metrics.keys())
    dominated: set = set()
    for i, n1 in enumerate(names):
        m1 = metrics[n1]
        for n2 in names:
            if n1 == n2:
                continue
            m2 = metrics[n2]
            # n2 dominates n1 if it's no worse on AUC and no worse on EER
            if m2.auc >= m1.auc and m2.eer <= m1.eer:
                if m2.auc > m1.auc or m2.eer < m1.eer:
                    dominated.add(n1)
                    break

    surviving = [n for n in names if n not in dominated]
    print(f"\n  Stage 4 Pareto filter: {len(names)} → {len(surviving)}")
    for n in names:
        tag = "✓" if n not in dominated else "✗ dominated"
        m = metrics[n]
        print(f"    {tag} {n}: AUC={m.auc:.4f}  EER={m.eer:.4f}")
    return surviving


# ------------------------------------------------------------------ #
#  Full pipeline                                                        #
# ------------------------------------------------------------------ #

def select_classifiers(
    csv_dir: str,
    label_csv: str = "",
    min_auc: float = 0.55,
    max_corr: float = 0.90,
    output_json: str = "",
    names: Optional[List[str]] = None,
) -> List[str]:
    """
    Run the full 4-stage weak classifier selection pipeline.

    Parameters
    ----------
    csv_dir    : directory containing {visual,rppg,sync}_scores.csv
    label_csv  : optional; if omitted, labels are read from the CSV files directly
    min_auc    : Stage 1 threshold (default 0.55)
    max_corr   : Stage 2 correlation cutoff (default 0.90)
    output_json: if set, write selected names + per-stage metrics to JSON
    names      : classifier names to load (default: ["visual", "rppg", "sync"])

    Returns
    -------
    List of selected classifier names (Pareto-surviving, non-redundant, AUC ≥ threshold).
    """
    print("=" * 60)
    print("Weak Classifier Selection Pipeline")
    print("=" * 60)

    data = _load_all_csvs(csv_dir, names)
    if not data:
        print("No valid CSV data found.")
        return []

    # Stage 1
    print("\nStage 1: AUC filter")
    data, all_metrics = filter_by_auc(data, min_auc)
    if not data:
        print("No classifiers passed AUC filter.")
        return []

    # Stage 2
    print("\nStage 2: Correlation filter")
    data = filter_by_correlation(data, all_metrics, max_corr)
    if not data:
        print("No classifiers survived correlation filter.")
        return []

    # Stage 3 (informational)
    compute_far_frr_curves(data)

    # Stage 4
    stage4_metrics = {n: all_metrics[n] for n in data if n in all_metrics}
    selected = pareto_filter(stage4_metrics)

    print(f"\n✓ Final selected classifiers: {selected}")

    if output_json:
        result = {
            "selected": selected,
            "all_metrics": {
                n: {"auc": m.auc, "eer": m.eer, "acc": m.acc}
                for n, m in all_metrics.items()
            },
            "min_auc": min_auc,
            "max_corr": max_corr,
        }
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote selection report → {output_json}")

    return selected
