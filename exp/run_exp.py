#!/usr/bin/env python3
"""
Experiment runner: test XceptionNet, TS-CAN, and Sync detectors
on FaceForensics++ and DeepFakeBench datasets.

Quick start
-----------
# Single detector, single dataset:
python exp/run_exp.py \\
    --detector xception \\
    --dataset ff \\
    --ff_root /data/FF++ \\
    --compression c23 \\
    --split test \\
    --max_videos 200

# Run all 6 combos (3 detectors × 2 dataset families):
python exp/run_exp.py \\
    --detector all \\
    --dataset all \\
    --ff_root /data/FF++ \\
    --celebdf_root /data/Celeb-DF-v2 \\
    --dfdc_root /data/DFDC \\
    --max_videos 200

# After collecting results, print the comparison table:
python exp/report.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# Project root on path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Exp-local modules
EXP = Path(__file__).parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from exp.utils.device import get_device, describe_backends
from preprocessing import UnifiedFaceExtractor, AudioExtractor
from detectors.visual_detector import VisualDetector
from detectors.sync_detector import SyncDetector
from exp.detectors.tscan_detector import TSCANDetector
from datasets.ff_dataset import FaceForensicsDataset, VideoSample
from datasets.dfdc_dataset import DFDCDataset
from exp.datasets.celebdf_dataset import CelebDFDataset
from evaluation.metrics import compute_metrics, DetectionMetrics


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _merge(base: dict, override: dict) -> dict:
    """Shallow merge: override wins at top level."""
    out = dict(base)
    out.update(override)
    return out


def _default_config() -> dict:
    base_path = EXP / "configs" / "base.yaml"
    if base_path.exists():
        return _load_yaml(str(base_path))
    return {}


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def _build_ff_samples(
    ff_root: str,
    compression: str,
    split: str,
    manipulation_types: Optional[List[str]],
    max_videos: Optional[int],
) -> Tuple[List[VideoSample], str]:
    ds = FaceForensicsDataset(
        ff_root,
        split=split,
        compression=compression,
        manipulation_types=manipulation_types,
    )
    samples = list(ds)
    if max_videos:
        # Balance real/fake when truncating
        real = [s for s in samples if s.label == 0][:max_videos // 2]
        fake = [s for s in samples if s.label == 1][:max_videos // 2]
        samples = real + fake

    stats = ds.stats()
    tag = f"FF++({compression}/{split}) real={stats.get('real',0)} fake={stats.get('fake',0)}"
    print(f"  Dataset: {tag}")
    return samples, f"ff_{compression}_{split}"


def _build_celebdf_samples(
    celebdf_root: str,
    version: str,
    split: str,
    max_videos: Optional[int],
) -> Tuple[List[VideoSample], str]:
    ds = CelebDFDataset(celebdf_root, version=version, split=split, max_videos=max_videos)
    stats = ds.stats()
    print(f"  Dataset: Celeb-DF-{version}({split}) real={stats['real']} fake={stats['fake']}")
    return list(ds), f"celebdf_{version}_{split}"


def _build_dfdc_samples(
    dfdc_root: str,
    split: str,
    max_videos: Optional[int],
) -> Tuple[List[VideoSample], str]:
    ds = DFDCDataset(dfdc_root, split=split, max_videos=max_videos)
    real = sum(1 for s in ds if s.label == 0)
    fake = sum(1 for s in ds if s.label == 1)
    print(f"  Dataset: DFDC({split}) real={real} fake={fake}")
    return list(ds), f"dfdc_{split}"


# ---------------------------------------------------------------------------
# Detector builders
# ---------------------------------------------------------------------------

def _build_visual_detector(cfg: dict) -> VisualDetector:
    vcfg = cfg.get("detectors", {}).get("visual", {})
    vcfg.setdefault("device", get_device(cfg.get("device", "auto")))
    return VisualDetector(vcfg)


def _build_tscan_detector(cfg: dict) -> TSCANDetector:
    tcfg = cfg.get("detectors", {}).get("tscan", {})
    tcfg.setdefault("device", cfg.get("device", "auto"))  # TSCANDetector resolves "auto"
    return TSCANDetector(tcfg)


def _build_sync_detector(cfg: dict) -> SyncDetector:
    scfg = cfg.get("detectors", {}).get("sync", {})
    scfg.setdefault("device", get_device(cfg.get("device", "auto")))
    return SyncDetector(scfg)


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    samples: List[VideoSample],
    detector_name: str,
    cfg: dict,
    max_videos: Optional[int],
    dataset_tag: str,
) -> DetectionMetrics:
    """Run one detector over samples and return metrics."""

    n = min(len(samples), max_videos) if max_videos else len(samples)
    samples = samples[:n]

    # Build extractor(s)
    pre_cfg = cfg.get("preprocessing", cfg)
    face_ext = UnifiedFaceExtractor(pre_cfg)
    audio_ext = AudioExtractor(pre_cfg) if detector_name == "sync" else None

    # Build detector
    if detector_name == "xception":
        det = _build_visual_detector(cfg)
    elif detector_name == "tscan":
        det = _build_tscan_detector(cfg)
    elif detector_name == "sync":
        det = _build_sync_detector(cfg)
    else:
        raise ValueError(f"Unknown detector: {detector_name}")

    labels: List[int] = []
    scores: List[Optional[float]] = []
    t0 = time.time()

    for i, sample in enumerate(samples):
        vid_name = Path(sample.video_path).name
        print(f"  [{i+1:>4}/{n}] {vid_name:<40} label={sample.label}", end="  ")

        try:
            face_track = face_ext.extract(sample.video_path)
        except Exception as e:
            print(f"SKIP (face extraction error: {e})")
            continue

        audio_path: Optional[str] = None
        if audio_ext is not None:
            try:
                audio_path = audio_ext.extract_to_temp(sample.video_path)
            except Exception:
                pass

        try:
            score = det.detect(face_track, audio_path)
        except Exception as e:
            print(f"SKIP (detector error: {e})")
            score = None

        if audio_path and Path(audio_path).exists():
            try:
                os.unlink(audio_path)
            except OSError:
                pass

        labels.append(sample.label)
        scores.append(score)
        tag = f"{score:.4f}" if score is not None else "None"
        print(f"score={tag}")

    elapsed = time.time() - t0
    print(f"\n  Processed {len(labels)} videos in {elapsed:.1f}s "
          f"({elapsed / max(len(labels), 1):.2f}s/video)")

    metrics = compute_metrics(
        labels, scores,
        dataset=dataset_tag,
        detector=detector_name,
    )
    return metrics


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def _save_result(metrics: DetectionMetrics, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "detector": metrics.detector,
        "dataset": metrics.dataset,
        "auc": metrics.auc,
        "acc": metrics.acc,
        "eer": metrics.eer,
        "ap": metrics.ap,
        "threshold": metrics.threshold,
        "n_real": metrics.n_real,
        "n_fake": metrics.n_fake,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ADB experiment runner: XceptionNet / TS-CAN / Sync on FF++ & DFB"
    )

    # Detector
    p.add_argument(
        "--detector",
        choices=["xception", "tscan", "sync", "all"],
        default="all",
        help="Which detector to evaluate (default: all)",
    )

    # Dataset family
    p.add_argument(
        "--dataset",
        choices=["ff", "dfb", "all"],
        default="all",
        help="Dataset family: ff (FF++ only) | dfb (all DFB sub-sets) | all",
    )

    # Data roots (any unused root is silently skipped)
    p.add_argument("--ff_root", default="", help="FF++ data root directory")
    p.add_argument("--celebdf_root", default="", help="Celeb-DF data root directory")
    p.add_argument("--dfdc_root", default="", help="DFDC data root directory")

    # FF++ specific
    p.add_argument("--compression", default="c23", choices=["c0", "c23", "c40"])
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument(
        "--manipulation_types",
        nargs="+",
        default=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
    )
    p.add_argument("--celebdf_version", default="v2", choices=["v1", "v2"])

    # Run control
    p.add_argument("--max_videos", type=int, default=None,
                   help="Max videos per (detector, dataset) pair")
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help=(
            "Compute device. 'auto' → cuda > mps (Apple Silicon) > cpu. "
            "'mps' runs PyTorch on Metal; rPPG signal ops use MLX when installed."
        ),
    )
    p.add_argument("--config", default="", help="Optional YAML config override")

    # Output
    p.add_argument(
        "--output_dir",
        default=str(EXP / "results"),
        help="Directory to write per-run JSON results",
    )
    return p.parse_args()


def _detector_names(args: argparse.Namespace) -> List[str]:
    return ["xception", "tscan", "sync"] if args.detector == "all" else [args.detector]


def _dataset_specs(args: argparse.Namespace):
    """
    Yield (samples, dataset_tag) for each requested dataset.
    Skips any dataset whose root is not provided.
    """
    include_ff = args.dataset in ("ff", "all")
    include_dfb = args.dataset in ("dfb", "all")

    if include_ff and args.ff_root:
        yield _build_ff_samples(
            args.ff_root, args.compression, args.split,
            args.manipulation_types, args.max_videos,
        )

    if include_dfb:
        if args.ff_root:
            yield _build_ff_samples(
                args.ff_root, args.compression, args.split,
                args.manipulation_types, args.max_videos,
            )
        if args.celebdf_root:
            yield _build_celebdf_samples(
                args.celebdf_root, args.celebdf_version, "test", args.max_videos,
            )
        if args.dfdc_root:
            yield _build_dfdc_samples(args.dfdc_root, "train", args.max_videos)

    if not args.ff_root and not args.celebdf_root and not args.dfdc_root:
        print("[WARNING] No data roots provided. Pass --ff_root / --celebdf_root / --dfdc_root.")


def main() -> None:
    args = parse_args()

    # Report available compute backends
    print("── Compute backends ──────────────────────")
    print(describe_backends())
    print("──────────────────────────────────────────\n")

    # Build config
    cfg = _default_config()
    if args.config and Path(args.config).exists():
        cfg = _merge(cfg, _load_yaml(args.config))
    cfg["device"] = args.device   # "auto" | "cuda" | "mps" | "cpu"

    detectors = _detector_names(args)
    dataset_iter = list(_dataset_specs(args))

    if not dataset_iter:
        print("No datasets to evaluate. Exiting.")
        return

    all_metrics: List[DetectionMetrics] = []

    for det_name in detectors:
        for samples, ds_tag in dataset_iter:
            # Deduplicate: dfb generates ff++ too → skip if already saw exact same tag+det
            out_path = str(Path(args.output_dir) / f"{det_name}_{ds_tag}.json")
            if Path(out_path).exists():
                print(f"\n[SKIP] {det_name} × {ds_tag} — result already exists at {out_path}")
                continue

            print(f"\n{'='*60}")
            print(f"  Detector : {det_name.upper()}")
            print(f"  Dataset  : {ds_tag}")
            print(f"  Videos   : {len(samples)}")
            print(f"{'='*60}")

            metrics = evaluate(samples, det_name, cfg, args.max_videos, ds_tag)
            all_metrics.append(metrics)

            print(f"\n{metrics}")
            _save_result(metrics, out_path)

    # Summary table
    if all_metrics:
        print("\n" + "=" * 70)
        print(f"  {'DETECTOR':<12} {'DATASET':<30} {'AUC':>6} {'ACC':>6} {'EER':>6} {'AP':>6}")
        print("-" * 70)
        for m in all_metrics:
            print(
                f"  {m.detector:<12} {m.dataset:<30} "
                f"{m.auc:>6.4f} {m.acc:>6.4f} {m.eer:>6.4f} {m.ap:>6.4f}"
            )
        print("=" * 70)
        print(f"\nRun `python exp/report.py` to regenerate this table from saved JSONs.")


if __name__ == "__main__":
    main()
