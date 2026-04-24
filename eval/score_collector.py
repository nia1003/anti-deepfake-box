"""
Sweep module variants and collect per-video fake_scores.

CLI:
    python eval/score_collector.py --data_root /data/FF++ --output eval/module_scores.json

Outputs
-------
eval/module_scores.json  — {module_id: {vid_id: score}}
eval/module_labels.json  — {vid_id: 0|1}   (1=fake, 0=real)
eval/module_stats.json   — {module_id: {EER, Max_FAR_FRR, avg_time_ms, peak_mb}}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Optional

# ── module variant definitions ────────────────────────────────────────────────

MODULE_VARIANTS: Dict[str, dict] = {
    "visual_v1": {"max_frames": 8,  "batch_size": 4},
    "visual_v2": {"max_frames": 16, "batch_size": 4},   # default
    "visual_v3": {"max_frames": 32, "batch_size": 8},
    "rppg_v1":   {"snr_threshold": 1.0, "snr_scale": 1.0},
    "rppg_v2":   {"snr_threshold": 1.5, "snr_scale": 1.0},  # default
    "rppg_v3":   {"snr_threshold": 2.0, "snr_scale": 1.0},
    "sync_v1":   {"whisper_model": "tiny"},    # default / fastest
    "sync_v2":   {"whisper_model": "base"},
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _eer_and_max_far_frr(scores: Dict[str, float], labels: Dict[str, int]):
    """Return (EER, Max_FAR_FRR) for a single module's scores."""
    import numpy as np

    vid_ids = [v for v in scores if v in labels]
    s = np.array([scores[v] for v in vid_ids])
    y = np.array([labels[v] for v in vid_ids])

    thresholds = np.linspace(0.0, 1.0, 101)
    fake_mask = y == 1
    real_mask = y == 0
    total_fake = fake_mask.sum()
    total_real = real_mask.sum()

    best_eer = 1.0
    max_far_frr = 0.0

    for t in thresholds:
        # fake_score >= t → judged FAKE
        FAR = (s[fake_mask] < t).sum() / max(total_fake, 1)   # fake wrongly judged REAL
        FRR = (s[real_mask] >= t).sum() / max(total_real, 1)  # real wrongly judged FAKE
        max_far_frr = max(max_far_frr, FAR * FRR)
        if abs(FAR - FRR) < abs(best_eer - 0.5) * 2 + 0.01:
            best_eer = (FAR + FRR) / 2.0

    return float(best_eer), float(max_far_frr)


def _build_visual_config(variant_params: dict, base_config: dict) -> dict:
    cfg = dict(base_config.get("detectors", {}).get("visual", {}))
    cfg.update(variant_params)
    return cfg


def _build_rppg_config(variant_params: dict, base_config: dict) -> dict:
    cfg = dict(base_config.get("detectors", {}).get("rppg", {}))
    cfg.update(variant_params)
    return cfg


def _build_sync_config(variant_params: dict, base_config: dict) -> dict:
    cfg = dict(base_config.get("detectors", {}).get("sync", {}))
    cfg.update(variant_params)
    return cfg


# ── dataset discovery ─────────────────────────────────────────────────────────

def _find_videos(data_root: Path, split: str = "test") -> Dict[str, int]:
    """
    Returns {vid_path: label} for a FaceForensics++ layout:
        data_root/
            real/<split>/*.mp4   (label=0)
            fake/<split>/*.mp4   (label=1)
    Also handles flat layout:
        data_root/
            manipulated_sequences/<method>/c23/videos/*.mp4  → fake
            original_sequences/youtube/c23/videos/*.mp4       → real
    """
    labels: Dict[str, int] = {}

    # simple two-folder layout
    for lbl, folder in [(0, "real"), (1, "fake")]:
        p = data_root / folder / split
        if p.exists():
            for f in sorted(p.glob("*.mp4")):
                labels[str(f)] = lbl

    # FF++ layout
    real_root = data_root / "original_sequences" / "youtube" / "c23" / "videos"
    if real_root.exists():
        for f in sorted(real_root.glob("*.mp4")):
            labels[str(f)] = 0
    for method_dir in (data_root / "manipulated_sequences").glob("*"):
        vid_dir = method_dir / "c23" / "videos"
        if vid_dir.exists():
            for f in sorted(vid_dir.glob("*.mp4")):
                labels[str(f)] = 1

    return labels


# ── main collection loop ──────────────────────────────────────────────────────

def collect_scores(
    data_root: Path,
    output_dir: Path,
    split: str = "test",
    config_path: Optional[Path] = None,
    variants: Optional[List[str]] = None,
) -> None:
    import yaml

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base config
    base_config: dict = {}
    if config_path and config_path.exists():
        with open(config_path) as f:
            base_config = yaml.safe_load(f) or {}

    # Discover videos
    vid_labels = _find_videos(data_root, split)
    if not vid_labels:
        print(f"[score_collector] No videos found under {data_root}", file=sys.stderr)
        sys.exit(1)
    print(f"[score_collector] Found {len(vid_labels)} videos ({sum(v==0 for v in vid_labels.values())} real, {sum(v==1 for v in vid_labels.values())} fake)")

    # Save labels
    labels_path = output_dir / "module_labels.json"
    with open(labels_path, "w") as f:
        json.dump(vid_labels, f, indent=2)
    print(f"[score_collector] Labels saved → {labels_path}")

    active_variants = variants or list(MODULE_VARIANTS.keys())

    # Load existing scores (allow resuming)
    scores_path = output_dir / "module_scores.json"
    all_scores: Dict[str, Dict[str, float]] = {}
    if scores_path.exists():
        with open(scores_path) as f:
            all_scores = json.load(f)

    stats_path = output_dir / "module_stats.json"
    all_stats: Dict[str, dict] = {}
    if stats_path.exists():
        with open(stats_path) as f:
            all_stats = json.load(f)

    # Import detectors (lazy; errors surface as warnings)
    sys.path.insert(0, str(Path(__file__).parent.parent))

    for variant_id in active_variants:
        if variant_id in all_scores:
            print(f"[score_collector] {variant_id}: already collected, skipping.")
            continue

        params = MODULE_VARIANTS[variant_id]
        prefix = variant_id.split("_v")[0]   # "visual", "rppg", "sync"

        print(f"\n[score_collector] === {variant_id} ===")
        vid_scores: Dict[str, float] = {}
        timing: List[float] = []
        peak_mb_list: List[float] = []

        for vid_path_str, _ in vid_labels.items():
            vid_path = Path(vid_path_str)
            if not vid_path.exists():
                continue

            try:
                tracemalloc.start()
                t0 = time.perf_counter()

                if prefix == "visual":
                    from detectors.visual_detector import VisualDetector
                    cfg = _build_visual_config(params, base_config)
                    cfg.setdefault("pretrained", str(Path("checkpoints/xception_ff_c23.pth")))
                    det = VisualDetector(cfg)
                    det.load()
                    from preprocessing.face_extractor import UnifiedFaceExtractor
                    extractor = UnifiedFaceExtractor(base_config.get("preprocessing", {}))
                    track = extractor.extract(vid_path)
                    score = det.predict(track)

                elif prefix == "rppg":
                    from detectors.rppg_detector import RPPGDetector
                    cfg = _build_rppg_config(params, base_config)
                    det = RPPGDetector(cfg)
                    det.load()
                    from preprocessing.face_extractor import UnifiedFaceExtractor
                    extractor = UnifiedFaceExtractor(base_config.get("preprocessing", {}))
                    track = extractor.extract(vid_path)
                    score = det.predict(track)

                elif prefix == "sync":
                    from detectors.sync_detector import SyncDetector
                    cfg = _build_sync_config(params, base_config)
                    det = SyncDetector(cfg)
                    det.load()
                    from preprocessing.face_extractor import UnifiedFaceExtractor
                    extractor = UnifiedFaceExtractor(base_config.get("preprocessing", {}))
                    track = extractor.extract(vid_path)
                    score = det.predict(track)

                else:
                    score = None

                dt = (time.perf_counter() - t0) * 1000  # ms
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                if score is not None:
                    vid_scores[vid_path_str] = float(score)
                    timing.append(dt)
                    peak_mb_list.append(peak / 1e6)

            except Exception as e:
                tracemalloc.stop()
                print(f"  [warn] {vid_path.name}: {e}")

        all_scores[variant_id] = vid_scores

        # Compute stats
        eer, max_far_frr = _eer_and_max_far_frr(vid_scores, vid_labels)
        all_stats[variant_id] = {
            "EER": eer,
            "Max_FAR_FRR": max_far_frr,
            "avg_time_ms": float(sum(timing) / len(timing)) if timing else 0.0,
            "peak_mb": float(sum(peak_mb_list) / len(peak_mb_list)) if peak_mb_list else 0.0,
            "n_videos": len(vid_scores),
        }
        print(f"  EER={eer:.4f}  Max_FAR×FRR={max_far_frr:.6f}  avg_time={all_stats[variant_id]['avg_time_ms']:.0f}ms")

        # Save incrementally
        with open(scores_path, "w") as f:
            json.dump(all_scores, f, indent=2)
        with open(stats_path, "w") as f:
            json.dump(all_stats, f, indent=2)

    print(f"\n[score_collector] Done. Scores → {scores_path}")
    print(f"[score_collector] Stats  → {stats_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Collect per-module fake scores for FAR/FRR analysis")
    ap.add_argument("--data_root",  type=Path, required=True,  help="Root of FF++ or compatible dataset")
    ap.add_argument("--output",     type=Path, default=Path("eval"), help="Output directory")
    ap.add_argument("--config",     type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--split",      default="test")
    ap.add_argument("--variants",   nargs="*", help="Subset of module variants to collect")
    args = ap.parse_args()

    collect_scores(
        data_root=args.data_root,
        output_dir=args.output,
        split=args.split,
        config_path=args.config,
        variants=args.variants,
    )


if __name__ == "__main__":
    main()
