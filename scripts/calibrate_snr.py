#!/usr/bin/env python3
"""
Calibrate rPPG SNR threshold on FF++ validation split.

Usage
-----
python scripts/calibrate_snr.py \
    --data_root /data/FF++ \
    --config configs/default.yaml \
    --split val \
    --output calibration_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing import UnifiedFaceExtractor
from detectors import RPPGDetector
from datasets.ff_dataset import FaceForensicsDataset
from evaluation.snr_calibration import calibrate_snr_threshold, update_config_threshold


def main():
    parser = argparse.ArgumentParser(description="Calibrate rPPG SNR threshold")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--compression", default="c23")
    parser.add_argument("--output", default="")
    parser.add_argument("--update_config", action="store_true",
                        help="Write calibrated threshold back to config YAML")
    parser.add_argument("--max_videos", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ds = FaceForensicsDataset(args.data_root, split=args.split, compression=args.compression)
    print(f"Calibrating on {args.split} split: {len(ds)} samples  {ds.stats()}")

    face_extractor = UnifiedFaceExtractor(config.get("preprocessing", config))
    rppg_det = RPPGDetector(config.get("detectors", {}).get("rppg", config))
    rppg_det.load()
    rppg_det._loaded = True

    snr_values, labels = [], []
    n = min(len(ds), args.max_videos) if args.max_videos else len(ds)

    for i, sample in enumerate(list(ds)[:n]):
        print(f"  [{i+1}/{n}] {Path(sample.video_path).name}", end="  ")
        face_track = face_extractor.extract(sample.video_path)
        if face_track is None:
            print("(no face, skipped)")
            continue
        info = rppg_det.get_ppg_and_snr(face_track)
        snr = info["snr_db"]
        snr_values.append(snr)
        labels.append(sample.label)
        print(f"SNR={snr:.2f} dB  label={'fake' if sample.label else 'real'} ({info['method']})")

    if not snr_values:
        print("No valid SNR values collected.")
        sys.exit(1)

    result = calibrate_snr_threshold(snr_values, labels)
    print("\n" + "=" * 60)
    print(f"Optimal SNR threshold : {result['snr_threshold']:.4f} dB")
    print(f"Youden's J statistic  : {result['j_statistic']:.4f}")
    print(f"Mean SNR (real)       : {result['snr_mean_real']:.4f} dB")
    print(f"Mean SNR (fake)       : {result['snr_mean_fake']:.4f} dB")
    print(f"n_real={result['n_real']}, n_fake={result['n_fake']}")
    print("=" * 60)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Calibration results saved to {args.output}")

    if args.update_config:
        update_config_threshold(args.config, result["snr_threshold"])


if __name__ == "__main__":
    main()
