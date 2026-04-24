"""
Parallel weighted fusion engine.

Runs a grid search over (alpha, beta, gamma) where:
    final_score = alpha * visual + beta * rppg + (1 - alpha - beta) * sync
    decision    = FAKE if final_score >= gamma else REAL

Uses pre-collected per-video scores (all_scores dict) — no re-inference.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np


# Grid definitions (matching the plan spec)
_ALPHA_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
_BETA_GRID  = [0.1, 0.2, 0.3, 0.4, 0.5]
_GAMMA_GRID = [0.3, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def _auc_score(true_labels, scores):
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(true_labels, scores)
    except Exception:
        return float("nan")


class ParallelFusion:
    """
    Weighted parallel fusion with grid-search over alpha/beta/gamma.

    Parameters
    ----------
    module_ids : list of module identifiers to include (order: visual, rppg, sync)
    alpha_grid, beta_grid, gamma_grid : override default grids (optional)
    """

    def __init__(
        self,
        module_ids: List[str] = None,
        alpha_grid: List[float] = None,
        beta_grid: List[float] = None,
        gamma_grid: List[float] = None,
    ):
        self.module_ids = module_ids or ["visual_v2", "rppg_v2", "sync_v1"]
        self.alpha_grid = alpha_grid or _ALPHA_GRID
        self.beta_grid  = beta_grid  or _BETA_GRID
        self.gamma_grid = gamma_grid or _GAMMA_GRID

    def _build_score_matrix(
        self,
        all_scores: Dict[str, Dict[str, float]],
        vid_ids: List[str],
    ) -> np.ndarray:
        """Shape (N, 3) where columns are [visual, rppg, sync] scores."""
        visual_id, rppg_id, sync_id = self.module_ids[0], self.module_ids[1], self.module_ids[2]
        mat = np.full((len(vid_ids), 3), 0.5)  # default = uncertain
        for j, mid in enumerate([visual_id, rppg_id, sync_id]):
            src = all_scores.get(mid, {})
            for i, vid in enumerate(vid_ids):
                if vid in src:
                    mat[i, j] = src[vid]
        return mat

    def grid_search(
        self,
        all_scores: Dict[str, Dict[str, float]],
        labels: Dict[str, int],
        far_budget: float = 0.10,
    ) -> List[dict]:
        """
        Run full grid search; return list of result dicts sorted by AUC desc.

        Each dict has keys:
            config_id, mode, alpha, beta, gamma,
            system_FAR, system_FRR, auc, avg_stages_used
        """
        vid_ids = list(labels.keys())
        true_labels = np.array([labels[v] for v in vid_ids])
        fake_mask = true_labels == 1
        real_mask = true_labels == 0
        total_fake = fake_mask.sum()
        total_real = real_mask.sum()

        mat = self._build_score_matrix(all_scores, vid_ids)  # (N, 3)

        results = []
        cfg_counter = 0

        for alpha, beta in itertools.product(self.alpha_grid, self.beta_grid):
            if alpha + beta > 1.0 + 1e-9:
                continue
            gamma_coeff = 1.0 - alpha - beta
            if gamma_coeff < 0:
                continue

            # vectorised final score for all videos
            final_scores = alpha * mat[:, 0] + beta * mat[:, 1] + gamma_coeff * mat[:, 2]

            auc = _auc_score(true_labels, final_scores)

            for gamma in self.gamma_grid:
                decisions = final_scores >= gamma
                system_FAR = (decisions[fake_mask] == False).sum() / max(total_fake, 1)
                system_FRR = (decisions[real_mask] == True).sum()  / max(total_real, 1)

                results.append({
                    "config_id": f"parallel_{cfg_counter:05d}",
                    "mode": "parallel",
                    "stage_order": ",".join(self.module_ids),
                    "alpha": alpha,
                    "beta": beta,
                    "gamma_sync": gamma_coeff,
                    "final_threshold": gamma,
                    "system_FAR": float(system_FAR),
                    "system_FRR": float(system_FRR),
                    "auc": float(auc),
                    "avg_stages_used": 3,  # parallel always uses all 3
                })
                cfg_counter += 1

        results.sort(key=lambda r: r["auc"], reverse=True)
        return results

    def predict(
        self,
        scores: Dict[str, float],
        alpha: float,
        beta: float,
        gamma: float,
    ) -> Tuple[str, float]:
        """
        Single-video prediction given pre-chosen alpha/beta/gamma.

        Returns (decision, final_score)
        """
        visual_score = scores.get(self.module_ids[0], 0.5)
        rppg_score   = scores.get(self.module_ids[1], 0.5)
        sync_score   = scores.get(self.module_ids[2], 0.5)
        gamma_coeff  = 1.0 - alpha - beta
        final_score  = alpha * visual_score + beta * rppg_score + gamma_coeff * sync_score
        decision = "FAKE" if final_score >= gamma else "REAL"
        return decision, float(final_score)
