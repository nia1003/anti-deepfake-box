"""
Permutation search + Pareto analysis for serial and parallel fusion configs.

Steps
-----
1. Enumerate M=1,2,3 stage permutations from representative module versions
   (visual_v2, rppg_v2, sync_v1) — 15 permutations total.
2. Sort each serial permutation by ascending total inference time.
3. Grid-search dual thresholds (H, L, F) for serial configs.
4. Run ParallelFusion grid search for parallel configs.
5. Pareto analysis: config A dominates B iff FAR_A ≤ FAR_B AND FRR_A ≤ FRR_B.
6. Output CSV files + scatter plot.

CLI:
    python eval/engine_search.py
        [--scores eval/module_scores.json]
        [--labels eval/module_labels.json]
        [--stats  eval/module_stats.json]
        [--outdir eval]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── threshold grids ───────────────────────────────────────────────────────────

H_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
L_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
F_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

# Representative module for each modality (single version per modality for search)
REPRESENTATIVE = {
    "visual": "visual_v2",
    "rppg":   "rppg_v2",
    "sync":   "sync_v1",
}
MODALITIES = list(REPRESENTATIVE.keys())  # ["visual", "rppg", "sync"]


# ── Pareto analysis ───────────────────────────────────────────────────────────

def pareto_front(configs: List[dict]) -> List[dict]:
    """Return non-dominated configs (minimise FAR and FRR simultaneously)."""
    front = []
    for i, a in enumerate(configs):
        dominated = False
        for j, b in enumerate(configs):
            if i == j:
                continue
            if (b["system_FAR"] <= a["system_FAR"] and
                    b["system_FRR"] <= a["system_FRR"] and
                    (b["system_FAR"] < a["system_FAR"] or b["system_FRR"] < a["system_FRR"])):
                dominated = True
                break
        if not dominated:
            front.append(a)
    return front


# ── serial config search ──────────────────────────────────────────────────────

def _serial_configs_for_order(
    stage_order: List[str],
    all_scores: dict,
    labels: dict,
    module_stats: dict,
) -> List[dict]:
    from fusion.serial_fusion import serial_decision

    vid_ids = list(labels.keys())
    fake_vids = [v for v in vid_ids if labels[v] == 1]
    real_vids = [v for v in vid_ids if labels[v] == 0]
    n = len(stage_order)
    results = []
    cfg_counter = [0]

    # total time (ms) for this permutation
    total_time_ms = sum(
        module_stats.get(m, {}).get("avg_time_ms", 0.0)
        for m in stage_order
    )

    def _run_search(H_vals, L_vals, F):
        # constraint: all Li < Hi
        if any(L_vals[k] >= H_vals[k] for k in range(len(H_vals))):
            return
        config = {
            "config_id": f"serial_{len(stage_order)}_{cfg_counter[0]:06d}",
            "stage_order": stage_order,
            "high_thresholds": list(H_vals),
            "low_thresholds":  list(L_vals),
            "final_threshold": F,
        }
        cfg_counter[0] += 1

        # empirical FAR/FRR
        missed_fake = 0
        missed_real = 0
        stages_sum = 0
        from sklearn.metrics import roc_auc_score
        agg_scores = []
        agg_labels = []

        for vid_id in vid_ids:
            sd = {m: all_scores[m][vid_id] for m in stage_order if vid_id in all_scores.get(m, {})}
            res = serial_decision(sd, config)
            stages_sum += res.stages_used
            agg = (sum(res.score_per_stage.values()) / len(res.score_per_stage)
                   if res.score_per_stage else 0.5)
            agg_scores.append(agg)
            agg_labels.append(labels[vid_id])

        for vid_id in fake_vids:
            sd = {m: all_scores[m][vid_id] for m in stage_order if vid_id in all_scores.get(m, {})}
            if serial_decision(sd, config).decision == "REAL":
                missed_fake += 1
        for vid_id in real_vids:
            sd = {m: all_scores[m][vid_id] for m in stage_order if vid_id in all_scores.get(m, {})}
            if serial_decision(sd, config).decision == "FAKE":
                missed_real += 1

        sys_far = missed_fake / max(len(fake_vids), 1)
        sys_frr = missed_real / max(len(real_vids), 1)
        avg_stages = stages_sum / max(len(vid_ids), 1)

        try:
            auc = float(roc_auc_score(agg_labels, agg_scores))
        except Exception:
            auc = float("nan")

        # estimate time (serial can exit early → weighted by avg_stages)
        est_time = (avg_stages / n) * total_time_ms

        results.append({
            "config_id": config["config_id"],
            "mode": "serial",
            "stage_order": ",".join(stage_order),
            "n_stages": n,
            "H": ",".join(str(h) for h in H_vals),
            "L": ",".join(str(l) for l in L_vals),
            "final_threshold": F,
            "system_FAR": sys_far,
            "system_FRR": sys_frr,
            "auc": auc,
            "avg_stages_used": avg_stages,
            "total_time_ms": est_time,
        })

    if n == 1:
        for F in F_GRID:
            _run_search([], [], F)
    elif n == 2:
        for H0 in H_GRID:
            for L0 in L_GRID:
                for F in F_GRID:
                    _run_search([H0], [L0], F)
    else:  # n == 3
        for H0, H1 in itertools.product(H_GRID, H_GRID):
            for L0, L1 in itertools.product(L_GRID, L_GRID):
                for F in F_GRID:
                    _run_search([H0, H1], [L0, L1], F)

    return results


# ── main search ───────────────────────────────────────────────────────────────

def run_pareto_search(
    scores_path: Path,
    labels_path: Path,
    stats_path: Path,
    output_dir: Path,
) -> Tuple[List[dict], List[dict]]:
    """
    Returns (all_configs, pareto_configs).
    Also writes CSV files and the Pareto scatter plot.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    with open(scores_path) as f:
        all_scores: Dict[str, Dict[str, float]] = json.load(f)
    with open(labels_path) as f:
        labels: Dict[str, int] = json.load(f)

    module_stats: dict = {}
    if stats_path.exists():
        with open(stats_path) as f:
            module_stats = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_configs: List[dict] = []

    # ── serial: M=1,2,3 permutations ─────────────────────────────────────────
    rep_modules = [REPRESENTATIVE[m] for m in MODALITIES]  # [visual_v2, rppg_v2, sync_v1]

    for m_count in [1, 2, 3]:
        for perm in itertools.permutations(rep_modules, m_count):
            # sort by ascending inference time (paper: put fastest first)
            sorted_perm = sorted(
                list(perm),
                key=lambda mid: module_stats.get(mid, {}).get("avg_time_ms", 9999),
            )
            print(f"[engine_search] Serial M={m_count} order={sorted_perm} ...", flush=True)
            cfgs = _serial_configs_for_order(sorted_perm, all_scores, labels, module_stats)
            all_configs.extend(cfgs)
            print(f"  {len(cfgs)} configs evaluated.")

    # ── parallel grid search ──────────────────────────────────────────────────
    print("[engine_search] Parallel grid search ...", flush=True)
    from fusion.parallel_fusion import ParallelFusion
    pf = ParallelFusion(module_ids=rep_modules)
    parallel_cfgs = pf.grid_search(all_scores, labels)
    # Compute total time for parallel (all 3 modules always run)
    parallel_time = sum(
        module_stats.get(m, {}).get("avg_time_ms", 0.0) for m in rep_modules
    )
    for c in parallel_cfgs:
        c["total_time_ms"] = parallel_time
        c["n_stages"] = 3
        c["H"] = ""
        c["L"] = ""
    all_configs.extend(parallel_cfgs)
    print(f"  {len(parallel_cfgs)} parallel configs evaluated.")

    # ── Pareto analysis ───────────────────────────────────────────────────────
    pareto_cfgs = pareto_front(all_configs)
    pareto_cfgs.sort(key=lambda c: c.get("total_time_ms", 0))

    # ── Write CSVs ────────────────────────────────────────────────────────────
    fieldnames = [
        "config_id", "mode", "stage_order", "n_stages",
        "H", "L", "final_threshold",
        "system_FAR", "system_FRR", "auc", "avg_stages_used", "total_time_ms",
    ]
    all_csv = output_dir / "all_engine_configs.csv"
    with open(all_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_configs)
    print(f"\n[engine_search] All configs → {all_csv} ({len(all_configs)} rows)")

    pareto_csv = output_dir / "pareto_configs.csv"
    with open(pareto_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(pareto_cfgs)
    print(f"[engine_search] Pareto configs → {pareto_csv} ({len(pareto_cfgs)} rows)")

    # ── Scatter plot ──────────────────────────────────────────────────────────
    _plot_pareto(all_configs, pareto_cfgs, output_dir / "pareto_plot.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_summary(all_configs, pareto_cfgs)

    return all_configs, pareto_cfgs


def _plot_pareto(all_configs, pareto_cfgs, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_far = [c["system_FAR"] for c in all_configs]
    all_frr = [c["system_FRR"] for c in all_configs]
    modes   = [c.get("mode", "serial") for c in all_configs]

    p_far = [c["system_FAR"] for c in pareto_cfgs]
    p_frr = [c["system_FRR"] for c in pareto_cfgs]

    fig, ax = plt.subplots(figsize=(9, 7))

    colors = {"serial": "tab:blue", "parallel": "tab:orange"}
    for mode_name in ["serial", "parallel"]:
        xs = [f for f, m in zip(all_far, modes) if m == mode_name]
        ys = [r for r, m in zip(all_frr, modes) if m == mode_name]
        ax.scatter(xs, ys, c=colors.get(mode_name, "gray"), alpha=0.25, s=8, label=mode_name)

    # Pareto front sorted by FAR for drawing
    sorted_p = sorted(zip(p_far, p_frr), key=lambda x: x[0])
    px_sorted, py_sorted = zip(*sorted_p) if sorted_p else ([], [])
    ax.plot(px_sorted, py_sorted, "r-o", linewidth=2, markersize=6, label="Pareto front", zorder=5)

    ax.set_xlabel("System FAR (fake → REAL miss rate)")
    ax.set_ylabel("System FRR (real → FAKE miss rate)")
    ax.set_title("Pareto Front: System FAR vs FRR\n(all serial + parallel configs)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120)
    plt.close(fig)
    print(f"[engine_search] Pareto plot → {out_path}")


def _print_summary(all_configs, pareto_cfgs):
    print("\n" + "=" * 60)
    print(f"Total configs tested : {len(all_configs)}")
    print(f"Pareto-optimal       : {len(pareto_cfgs)}")

    # Fastest serial config with FAR ≤ 0.05
    serial_cfgs = [c for c in all_configs if c.get("mode") == "serial"]
    acceptable = [c for c in serial_cfgs if c["system_FAR"] <= 0.05]
    if acceptable:
        fastest = min(acceptable, key=lambda c: c.get("total_time_ms", 9999))
        print(f"\nFastest serial (FAR≤0.05):")
        print(f"  {fastest['config_id']}  order={fastest['stage_order']}")
        print(f"  FAR={fastest['system_FAR']:.4f}  FRR={fastest['system_FRR']:.4f}"
              f"  time={fastest.get('total_time_ms', 0):.0f}ms")
    else:
        print("\nNo serial config with FAR ≤ 0.05 found.")

    # Highest AUC
    best_auc = max(all_configs, key=lambda c: c.get("auc") or 0, default=None)
    if best_auc:
        print(f"\nHighest AUC config:")
        print(f"  {best_auc['config_id']}  mode={best_auc.get('mode')}  order={best_auc.get('stage_order')}")
        print(f"  AUC={best_auc.get('auc', float('nan')):.4f}"
              f"  FAR={best_auc['system_FAR']:.4f}  FRR={best_auc['system_FRR']:.4f}")
    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Permutation + Pareto search over fusion configs")
    ap.add_argument("--scores", type=Path, default=Path("eval/module_scores.json"))
    ap.add_argument("--labels", type=Path, default=Path("eval/module_labels.json"))
    ap.add_argument("--stats",  type=Path, default=Path("eval/module_stats.json"))
    ap.add_argument("--outdir", type=Path, default=Path("eval"))
    args = ap.parse_args()

    run_pareto_search(
        scores_path=args.scores,
        labels_path=args.labels,
        stats_path=args.stats,
        output_dir=args.outdir,
    )


if __name__ == "__main__":
    main()
