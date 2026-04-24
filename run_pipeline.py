"""
Anti-Deepfake-Box — Two-mode pipeline entry point.

Modes
-----
online  : real-time streaming detection via serial cascade
          (fastest Pareto config with FAR ≤ 0.05, 5 s sliding windows)
offline : post-hoc forensic analysis via parallel fusion
          (highest-AUC Pareto config, full video)

Usage
-----
    python run_pipeline.py --video call.mp4 --mode online
    python run_pipeline.py --video evidence.mp4 --mode offline

Falls back to WeightedEnsemble if eval/pareto_configs.csv does not exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
PARETO_CSV = ROOT / "eval" / "pareto_configs.csv"
RESULTS_DIR = ROOT / "results"


# ── CSV loader ────────────────────────────────────────────────────────────────

def _load_pareto_configs() -> List[dict]:
    if not PARETO_CSV.exists():
        return []
    with open(PARETO_CSV, newline="") as f:
        return list(csv.DictReader(f))


def _select_online_config(configs: List[dict]) -> Optional[dict]:
    """Fastest serial config with FAR ≤ 0.05."""
    serial = [c for c in configs if c.get("mode") == "serial"]
    acceptable = [c for c in serial if float(c.get("system_FAR", 1)) <= 0.05]
    if not acceptable:
        return None
    return min(acceptable, key=lambda c: float(c.get("total_time_ms", 9999)))


def _select_offline_config(configs: List[dict]) -> Optional[dict]:
    """Highest-AUC config (any mode)."""
    valid = [c for c in configs if c.get("auc") not in (None, "", "nan")]
    if not valid:
        return None
    return max(valid, key=lambda c: float(c.get("auc", 0)))


# ── face + detector loading ───────────────────────────────────────────────────

def _load_detectors(cfg_yaml: dict):
    from detectors.visual_detector import VisualDetector
    from detectors.rppg_detector import RPPGDetector
    from detectors.sync_detector import SyncDetector

    detectors = {}
    det_cfg = cfg_yaml.get("detectors", {})

    vis = VisualDetector(det_cfg.get("visual", {}))
    vis.load()
    detectors["visual_v2"] = vis

    rppg = RPPGDetector(det_cfg.get("rppg", {}))
    rppg.load()
    detectors["rppg_v2"] = rppg

    sync = SyncDetector(det_cfg.get("sync", {}))
    sync.load()
    detectors["sync_v1"] = sync

    return detectors


def _infer_scores(detectors: dict, track, audio_path: Optional[Path]) -> Dict[str, float]:
    scores = {}
    for mid, det in detectors.items():
        try:
            if mid.startswith("sync") and audio_path is not None:
                s = det.predict(track, audio_path=audio_path)
            else:
                s = det.predict(track)
            if s is not None:
                scores[mid] = float(s)
        except Exception as e:
            print(f"  [warn] {mid}: {e}", file=sys.stderr)
    return scores


# ── window splitting ──────────────────────────────────────────────────────────

def _split_windows(track, fps: float, window_sec: float = 5.0):
    """Yield sub-FaceTrack slices of length window_sec * fps frames."""
    from preprocessing.face_extractor import FaceTrack
    import numpy as np

    n_frames = len(track.aligned_frames)
    window_frames = max(1, int(window_sec * fps))
    n_windows = max(1, (n_frames + window_frames - 1) // window_frames)

    for i in range(n_windows):
        start = i * window_frames
        end = min(start + window_frames, n_frames)
        sub = FaceTrack(
            aligned_frames=track.aligned_frames[start:end],
            bboxes=track.bboxes[start:end] if track.bboxes is not None else None,
            landmarks=track.landmarks[start:end] if track.landmarks is not None else None,
            fps=track.fps,
        )
        yield i, n_windows, sub


# ── online mode ───────────────────────────────────────────────────────────────

def run_online(video_path: Path, cfg_yaml: dict, pareto_config: Optional[dict]) -> dict:
    from preprocessing.face_extractor import UnifiedFaceExtractor
    from fusion.serial_fusion import serial_decision, SerialFusion
    from fusion.weighted_ensemble import WeightedEnsemble

    print(f"\n[online] Analysing: {video_path.name}")
    print(f"[online] Mode: serial cascade | window: 5 s")

    extractor = UnifiedFaceExtractor(cfg_yaml.get("preprocessing", {}))
    track = extractor.extract(video_path)
    fps = track.fps

    # Build serial config from pareto_config row or fallback
    if pareto_config:
        stage_order = pareto_config["stage_order"].split(",")
        H_vals = [float(h) for h in pareto_config["H"].split(",")] if pareto_config.get("H") else []
        L_vals = [float(l) for l in pareto_config["L"].split(",")] if pareto_config.get("L") else []
        F = float(pareto_config.get("final_threshold", 0.5))
        config = {
            "config_id": pareto_config["config_id"],
            "stage_order": stage_order,
            "high_thresholds": H_vals,
            "low_thresholds": L_vals,
            "final_threshold": F,
        }
        use_fallback = False
    else:
        print("[online] No Pareto config found — using WeightedEnsemble fallback.")
        use_fallback = True

    detectors = _load_detectors(cfg_yaml)

    window_log = []
    for win_idx, n_wins, sub_track in _split_windows(track, fps):
        t0 = time.perf_counter()
        scores = _infer_scores(detectors, sub_track, None)
        dt = time.perf_counter() - t0

        if use_fallback:
            ens = WeightedEnsemble(cfg_yaml.get("fusion", {}))
            result = ens.fuse(scores)
            decision = result.prediction
            stages = len(scores)
        else:
            res = serial_decision(scores, config)
            decision = res.decision
            stages = res.stages_used

        tag = "FAKE" if decision == "FAKE" else "real"
        score_str = " | ".join(f"{k}:{v:.3f}" for k, v in scores.items())
        print(f"  [{win_idx+1:02d}/{n_wins}] {score_str} | stages:{stages} | {tag}  ({dt*1000:.0f}ms)")

        window_log.append({
            "window": win_idx + 1,
            "decision": decision,
            "scores": scores,
            "stages_used": stages,
        })

    fake_count = sum(1 for w in window_log if w["decision"] == "FAKE")
    fake_ratio = fake_count / max(len(window_log), 1)
    final_decision = "FAKE" if fake_ratio >= 0.5 else "REAL"

    print(f"\n[online] Result: {final_decision}  (fake windows: {fake_count}/{len(window_log)} = {fake_ratio:.0%})")

    output = {
        "video": str(video_path),
        "mode": "online",
        "windows": window_log,
        "fake_ratio": fake_ratio,
        "final_decision": final_decision,
        "config_id": pareto_config["config_id"] if pareto_config else "fallback",
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    log_path = RESULTS_DIR / f"call_log_{video_path.stem}.json"
    with open(log_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[online] Log saved → {log_path}")
    return output


# ── offline mode ──────────────────────────────────────────────────────────────

def run_offline(video_path: Path, cfg_yaml: dict, pareto_config: Optional[dict]) -> dict:
    import tempfile
    import subprocess
    from preprocessing.face_extractor import UnifiedFaceExtractor
    from fusion.parallel_fusion import ParallelFusion
    from fusion.weighted_ensemble import WeightedEnsemble

    print(f"\n[offline] Analysing: {video_path.name}")
    print(f"[offline] Mode: parallel weighted fusion | full video")

    extractor = UnifiedFaceExtractor(cfg_yaml.get("preprocessing", {}))
    track = extractor.extract(video_path)

    # Extract audio
    audio_path = None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ar", "16000", "-ac", "1", tmp.name],
            capture_output=True, check=True,
        )
        audio_path = Path(tmp.name)
    except Exception:
        pass

    detectors = _load_detectors(cfg_yaml)
    scores = _infer_scores(detectors, track, audio_path)

    if pareto_config and pareto_config.get("mode") == "parallel":
        alpha = float(pareto_config.get("alpha", 0.5))
        beta  = float(pareto_config.get("beta", 0.25))
        gamma = float(pareto_config.get("final_threshold", 0.5))
        pf = ParallelFusion(module_ids=list(scores.keys()))
        decision, final_score = pf.predict(scores, alpha, beta, gamma)
        cfg_id = pareto_config["config_id"]
    else:
        print("[offline] No parallel Pareto config — using WeightedEnsemble fallback.")
        ens = WeightedEnsemble(cfg_yaml.get("fusion", {}))
        result = ens.fuse(scores)
        decision = result.prediction
        final_score = result.fake_score
        cfg_id = "fallback"

    print(f"\n[offline] Scores: {scores}")
    print(f"[offline] Final score: {final_score:.4f}  → {decision}")

    output = {
        "video": str(video_path),
        "mode": "offline",
        "scores": scores,
        "final_score": final_score,
        "decision": decision,
        "config_id": cfg_id,
        "pareto_config": pareto_config,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    report_path = RESULTS_DIR / f"forensic_report_{video_path.stem}.json"
    with open(report_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[offline] Report saved → {report_path}")
    return output


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Anti-Deepfake-Box — two-mode inference pipeline"
    )
    ap.add_argument("--video",  type=Path, required=True, help="Input video file")
    ap.add_argument("--mode",   choices=["online", "offline"], default="online")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    args = ap.parse_args()

    if not args.video.exists():
        print(f"Error: video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    import yaml
    cfg_yaml: dict = {}
    if args.config.exists():
        with open(args.config) as f:
            cfg_yaml = yaml.safe_load(f) or {}

    pareto_configs = _load_pareto_configs()

    if args.mode == "online":
        cfg = _select_online_config(pareto_configs)
        if cfg is None and pareto_configs:
            # Relax: pick fastest serial regardless of FAR
            serial = [c for c in pareto_configs if c.get("mode") == "serial"]
            cfg = min(serial, key=lambda c: float(c.get("total_time_ms", 9999))) if serial else None
        run_online(args.video, cfg_yaml, cfg)
    else:
        cfg = _select_offline_config(pareto_configs)
        run_offline(args.video, cfg_yaml, cfg)


if __name__ == "__main__":
    main()
