"""
Serial cascade fusion with dual thresholds per intermediate stage.

BMMA-GPT reference:
  Dual-threshold biometric authentication → deepfake fake_score direction:
    fake_score >= H  → FAKE  (early-exit, high confidence fake)
    fake_score <= L  → REAL  (early-exit, high confidence real)
    L < score < H   → uncertain, pass to next stage
  Final stage: score >= F → FAKE; else REAL

DOI: 10.1109/TDSC.2025.3620382
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SerialResult:
    decision: str                     # "FAKE" | "REAL"
    stages_used: int
    score_per_stage: Dict[str, float] = field(default_factory=dict)
    early_exit_stage: int = -1        # 0-indexed; -1 = reached final stage
    config_id: str = ""


def serial_decision(scores_dict: dict, config: dict) -> SerialResult:
    """
    Run a single video through the serial cascade.

    Parameters
    ----------
    scores_dict : {module_id: score}  — only keys present in stage_order are used
    config : {
        "config_id": str,
        "stage_order": [m0, m1, ..., mN],       length N
        "high_thresholds": [H0, ..., H_{N-2}],  length N-1
        "low_thresholds":  [L0, ..., L_{N-2}],  length N-1  (each Li < Hi)
        "final_threshold": F,
    }
    """
    order: List[str] = config["stage_order"]
    H: List[float] = config["high_thresholds"]
    L: List[float] = config["low_thresholds"]
    F: float = config["final_threshold"]
    config_id: str = config.get("config_id", "")

    n = len(order)
    score_per_stage: Dict[str, float] = {}
    early_exit_stage = -1

    for i, module_id in enumerate(order):
        score = scores_dict.get(module_id)
        if score is None:
            # module score missing — treat as uncertain (score = 0.5 midpoint)
            score = 0.5
        score_per_stage[module_id] = score

        if i < n - 1:
            # intermediate stage: dual threshold
            if score >= H[i]:
                return SerialResult(
                    decision="FAKE",
                    stages_used=i + 1,
                    score_per_stage=score_per_stage,
                    early_exit_stage=i,
                    config_id=config_id,
                )
            if score <= L[i]:
                return SerialResult(
                    decision="REAL",
                    stages_used=i + 1,
                    score_per_stage=score_per_stage,
                    early_exit_stage=i,
                    config_id=config_id,
                )
            # uncertain → continue
        else:
            # final stage: single threshold
            decision = "FAKE" if score >= F else "REAL"
            return SerialResult(
                decision=decision,
                stages_used=n,
                score_per_stage=score_per_stage,
                early_exit_stage=-1,
                config_id=config_id,
            )

    # fallback (shouldn't be reached)
    return SerialResult(
        decision="REAL",
        stages_used=0,
        score_per_stage=score_per_stage,
        config_id=config_id,
    )


def compute_system_metrics(
    config: dict,
    all_scores: dict,
    labels: dict,
) -> dict:
    """
    Empirically evaluate a serial config over all videos.

    Parameters
    ----------
    config      : serial config dict (see serial_decision)
    all_scores  : {module_id: {vid_id: score}}
    labels      : {vid_id: 0|1}  (1 = fake, 0 = real)

    Returns
    -------
    {system_FAR, system_FRR, avg_stages_used, auc}
    """
    from sklearn.metrics import roc_auc_score

    fake_videos = [v for v, lbl in labels.items() if lbl == 1]
    real_videos = [v for v, lbl in labels.items() if lbl == 0]

    total_fake = len(fake_videos)
    total_real = len(real_videos)

    missed_fake = 0   # fake judged as REAL (false accept)
    missed_real = 0   # real judged as FAKE (false reject)
    stages_sum = 0

    all_vid_ids = list(labels.keys())
    final_scores: List[float] = []
    true_labels: List[int] = []

    for vid_id in all_vid_ids:
        scores_dict = {m: all_scores[m][vid_id] for m in config["stage_order"] if vid_id in all_scores.get(m, {})}
        result = serial_decision(scores_dict, config)
        stages_sum += result.stages_used
        # aggregate score: mean of stages used
        agg = (sum(result.score_per_stage.values()) / len(result.score_per_stage)
               if result.score_per_stage else 0.5)
        final_scores.append(agg)
        true_labels.append(labels[vid_id])

    for vid_id in fake_videos:
        scores_dict = {m: all_scores[m][vid_id] for m in config["stage_order"] if vid_id in all_scores.get(m, {})}
        result = serial_decision(scores_dict, config)
        if result.decision == "REAL":
            missed_fake += 1

    for vid_id in real_videos:
        scores_dict = {m: all_scores[m][vid_id] for m in config["stage_order"] if vid_id in all_scores.get(m, {})}
        result = serial_decision(scores_dict, config)
        if result.decision == "FAKE":
            missed_real += 1

    system_FAR = missed_fake / total_fake if total_fake > 0 else 0.0
    system_FRR = missed_real / total_real if total_real > 0 else 0.0
    avg_stages = stages_sum / len(all_vid_ids) if all_vid_ids else 0.0

    try:
        auc = roc_auc_score(true_labels, final_scores)
    except Exception:
        auc = float("nan")

    return {
        "system_FAR": system_FAR,
        "system_FRR": system_FRR,
        "avg_stages_used": avg_stages,
        "auc": auc,
    }


def compute_system_metrics_analytical(config: dict, module_stats: dict) -> dict:
    """
    Analytical approximation using per-module FAR/FRR stats (paper Eq.5/6).

    Uses con_b = Max(FAR × FRR) as the conservative bound per stage.
    Faster than empirical computation — no per-video iteration needed.

    Returns {system_FAR, system_FRR}  (auc / avg_stages not computable analytically)
    """
    order: List[str] = config["stage_order"]
    H: List[float] = config["high_thresholds"]
    L: List[float] = config["low_thresholds"]
    F: float = config["final_threshold"]
    n = len(order)

    # system_FAR = P(all intermediate stages pass fake through AND final says REAL)
    # Approximated as product of per-stage "pass-through" probabilities.
    # For a fake video at stage i (intermediate):
    #   P(not caught) = P(score < H[i]) ≈ FAR(H[i]) ... but we use con_b as bound.
    # Here we use the Max_FAR_FRR heuristic from the paper.

    sys_far = 1.0
    sys_frr = 1.0

    for i, module_id in enumerate(order):
        stats = module_stats.get(module_id, {})
        eer = stats.get("EER", 0.1)
        con_b = stats.get("Max_FAR_FRR", eer ** 2)

        if i < n - 1:
            # intermediate: dual threshold — con_b bounds the pass-through error
            sys_far *= math.sqrt(con_b)
            sys_frr *= math.sqrt(con_b)
        else:
            # final stage: single threshold at F (use EER as proxy for threshold F)
            sys_far *= eer
            sys_frr *= eer

    return {
        "system_FAR": sys_far,
        "system_FRR": sys_frr,
    }


class SerialFusion:
    """Thin class wrapper around serial_decision / compute_system_metrics."""

    def __init__(self, config: dict):
        self.config = config

    def decide(self, scores_dict: dict) -> SerialResult:
        return serial_decision(scores_dict, self.config)

    def evaluate(self, all_scores: dict, labels: dict) -> dict:
        return compute_system_metrics(self.config, all_scores, labels)

    def evaluate_analytical(self, module_stats: dict) -> dict:
        return compute_system_metrics_analytical(self.config, module_stats)
