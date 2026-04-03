#!/usr/bin/env python3
"""
Anti-Deepfake-Box: Dataset-level evaluation.

Usage
-----
# FF++ c23 evaluation:
python scripts/evaluate.py \
    --dataset ff++ \
    --data_root /data/FF++ \
    --compression c23 \
    --config configs/default.yaml \
    --split test \
    --output results/ff_test.json

# Custom real/fake dirs:
python scripts/evaluate.py \
    --real_dir /data/real_videos \
    --fake_dir /data/fake_videos \
    --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing import UnifiedFaceExtractor, AudioExtractor
from detectors import VisualDetector, RPPGDetector, SyncDetector
from fusion import WeightedEnsemble
from datasets.ff_dataset import FaceForensicsDataset, VideoSample
from datasets.video_dataset import VideoDataset
from evaluation.metrics import compute_metrics, DetectionMetrics


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate_samples(
    samples: List[VideoSample],
    config: dict,
    skip: list = None,
    max_videos: Optional[int] = None,
) -> DetectionMetrics:
    skip = skip or []

    face_extractor = UnifiedFaceExtractor(config.get("preprocessing", config))
    audio_extractor = AudioExtractor(config.get("preprocessing", config))

    visual_det = VisualDetector(config.get("detectors", {}).get("visual", config)) if "visual" not in skip else None
    rppg_det   = RPPGDetector(config.get("detectors", {}).get("rppg", config))    if "rppg" not in skip else None
    sync_det   = SyncDetector(config.get("detectors", {}).get("sync", config))    if "sync" not in skip else None

    fuser = WeightedEnsemble(config)

    labels: List[int] = []
    scores: List[Optional[float]] = []

    n = min(len(samples), max_videos) if max_videos else len(samples)
    t_start = time.time()

    for i, sample in enumerate(samples[:n]):
        print(f"  [{i+1}/{n}] {Path(sample.video_path).name}  label={sample.label}", end=" ")

        face_track = face_extractor.extract(sample.video_path)

        audio_path = None
        if sync_det is not None:
            audio_path = audio_extractor.extract_to_temp(sample.video_path)

        s_visual = visual_det.detect(face_track) if visual_det else None
        s_rppg   = rppg_det.detect(face_track)   if rppg_det else None
        s_sync   = sync_det.detect(face_track, audio_path) if sync_det else None

        result = fuser.fuse(s_visual, s_rppg, s_sync)

        # Cleanup temp audio
        if audio_path and Path(audio_path).exists():
            try:
                os.unlink(audio_path)
            except OSError:
                pass

        labels.append(sample.label)
        scores.append(result.fake_score)
        print(f"→ fake_score={result.fake_score:.4f} ({'FAKE' if result.is_fake else 'REAL'})")

    elapsed = time.time() - t_start
    print(f"\nEvaluated {n} videos in {elapsed:.1f}s ({elapsed/max(n,1):.2f}s/video)")

    metrics = compute_metrics(labels, scores)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Anti-Deepfake-Box dataset evaluation")
    parser.add_argument("--dataset", choices=["ff++", "custom"], default="ff++")
    parser.add_argument("--data_root", default="", help="FF++ data root")
    parser.add_argument("--compression", default="c23", choices=["c0", "c23", "c40"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--real_dir", default="", help="Custom real videos dir")
    parser.add_argument("--fake_dir", default="", help="Custom fake videos dir")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="", help="Save results to JSON")
    parser.add_argument("--skip", nargs="+", choices=["visual", "rppg", "sync"], default=[])
    parser.add_argument("--max_videos", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    if args.dataset == "ff++":
        if not args.data_root:
            parser.error("--data_root required for ff++ dataset")
        ds = FaceForensicsDataset(args.data_root, split=args.split, compression=args.compression)
        print(f"FF++ {args.split} ({args.compression}): {ds.stats()}")
    else:
        if not args.real_dir and not args.fake_dir:
            parser.error("--real_dir or --fake_dir required for custom dataset")
        ds = VideoDataset(real_dir=args.real_dir or None, fake_dir=args.fake_dir or None)

    print(f"\nEvaluating {len(ds)} videos...")
    metrics = evaluate_samples(list(ds), config, skip=args.skip, max_videos=args.max_videos)

    print("\n" + "=" * 60)
    print(metrics)
    print("=" * 60)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
