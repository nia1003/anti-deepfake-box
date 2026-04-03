#!/usr/bin/env python3
"""
Anti-Deepfake-Box: Single-video inference with async pipeline.

Usage
-----
# Sync (default):
python scripts/inference.py --video sample.mp4 --config configs/default.yaml

# Async (faster for long videos):
python scripts/inference.py --video sample.mp4 --config configs/default.yaml --async

# Skip specific modalities:
python scripts/inference.py --video sample.mp4 --skip sync

# Use meta-classifier fusion:
python scripts/inference.py --video sample.mp4 --fusion_mode meta
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing import UnifiedFaceExtractor, AudioExtractor
from detectors import VisualDetector, RPPGDetector, SyncDetector
from fusion import WeightedEnsemble, MetaClassifier
from fusion.weighted_ensemble import FusionResult


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_detectors(config: dict, skip: list = None):
    skip = skip or []
    detectors = {}
    if "visual" not in skip:
        detectors["visual"] = VisualDetector(config.get("detectors", {}).get("visual", config))
    if "rppg" not in skip:
        detectors["rppg"] = RPPGDetector(config.get("detectors", {}).get("rppg", config))
    if "sync" not in skip:
        detectors["sync"] = SyncDetector(config.get("detectors", {}).get("sync", config))
    return detectors


def build_fusion(config: dict):
    mode = config.get("fusion", {}).get("mode", "weighted")
    if mode == "meta":
        fuser = MetaClassifier(config)
        model_path = config.get("fusion", {}).get("meta", {}).get("model_path", "")
        if model_path and Path(model_path).exists():
            fuser.load()
        else:
            print("[WARNING] Meta-classifier weights not found, falling back to weighted ensemble.")
            fuser = WeightedEnsemble(config)
    else:
        fuser = WeightedEnsemble(config)
    return fuser


# ------------------------------------------------------------------ #
#  Synchronous pipeline                                                #
# ------------------------------------------------------------------ #

def detect_video(
    video_path: str,
    config: dict,
    skip: list = None,
) -> FusionResult:
    """Synchronous three-path detection + fusion."""
    face_extractor = UnifiedFaceExtractor(config.get("preprocessing", config))
    audio_extractor = AudioExtractor(config.get("preprocessing", config))
    detectors = build_detectors(config, skip)
    fuser = build_fusion(config)

    t0 = time.time()

    # Step 1: SSOT face extraction
    face_track = face_extractor.extract(video_path)
    if face_track is None:
        print(f"[WARNING] No face detected in {video_path}")
        face_track_available = False
    else:
        face_track_available = True
        print(f"  Face track: {face_track.T} frames @ {face_track.fps:.1f} fps "
              f"(detection: {time.time()-t0:.1f}s)")

    # Step 2: Audio extraction
    audio_path = None
    if "sync" in detectors:
        audio_path = audio_extractor.extract_to_temp(video_path)
        if audio_path:
            print(f"  Audio extracted: {audio_path}")

    # Step 3: Run three detectors
    scores = {}
    for name, detector in detectors.items():
        t1 = time.time()
        if not face_track_available:
            scores[name] = None
            continue
        score = detector.detect(face_track, audio_path if name == "sync" else None)
        scores[name] = score
        val = f"{score:.4f}" if score is not None else "N/A"
        print(f"  [{name:8s}] score={val}  ({time.time()-t1:.1f}s)")

    # Cleanup temp audio
    if audio_path and Path(audio_path).exists():
        try:
            os.unlink(audio_path)
        except OSError:
            pass

    # Step 4: Fusion
    result = fuser.fuse(
        visual_score=scores.get("visual"),
        rppg_score=scores.get("rppg"),
        sync_score=scores.get("sync"),
    )
    print(f"\nTotal inference time: {time.time()-t0:.1f}s")
    return result


# ------------------------------------------------------------------ #
#  Asynchronous pipeline (audio + face detection in parallel)         #
# ------------------------------------------------------------------ #

async def detect_video_async(
    video_path: str,
    config: dict,
    skip: list = None,
) -> FusionResult:
    """
    Async pipeline: audio extraction and face detection run concurrently.
    Three detectors also run concurrently (GPU-aware via executor).
    """
    face_extractor = UnifiedFaceExtractor(config.get("preprocessing", config))
    audio_extractor = AudioExtractor(config.get("preprocessing", config))
    detectors = build_detectors(config, skip)
    fuser = build_fusion(config)

    loop = asyncio.get_event_loop()
    t0 = time.time()

    # Parallel: face extraction + audio extraction
    face_task = loop.run_in_executor(None, face_extractor.extract, video_path)
    audio_task = loop.run_in_executor(
        None,
        audio_extractor.extract_to_temp,
        video_path,
    ) if "sync" in detectors else asyncio.sleep(0)

    face_track, audio_path = await asyncio.gather(face_task, audio_task)
    audio_path = audio_path if isinstance(audio_path, (str, type(None))) else None

    print(f"  SSOT+Audio extraction: {time.time()-t0:.1f}s")

    # Parallel: three detectors
    async def run_detector(name, detector):
        t1 = time.time()
        ap = audio_path if name == "sync" else None
        score = await detector.detect_async(face_track, ap)
        val = f"{score:.4f}" if score is not None else "N/A"
        print(f"  [{name:8s}] score={val}  ({time.time()-t1:.1f}s)")
        return name, score

    tasks = [run_detector(n, d) for n, d in detectors.items()]
    results = await asyncio.gather(*tasks)
    scores = dict(results)

    # Cleanup
    if audio_path and Path(audio_path).exists():
        try:
            os.unlink(audio_path)
        except OSError:
            pass

    result = fuser.fuse(
        visual_score=scores.get("visual"),
        rppg_score=scores.get("rppg"),
        sync_score=scores.get("sync"),
    )
    print(f"\nTotal async inference time: {time.time()-t0:.1f}s")
    return result


# ------------------------------------------------------------------ #
#  CLI entry point                                                     #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Anti-Deepfake-Box single-video inference")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--config", default="configs/default.yaml", help="Config YAML path")
    parser.add_argument("--fusion_mode", choices=["weighted", "meta"], default=None,
                        help="Override fusion mode from config")
    parser.add_argument("--skip", nargs="+", choices=["visual", "rppg", "sync"],
                        default=[], help="Skip specified modalities")
    parser.add_argument("--async_mode", action="store_true",
                        help="Use async pipeline (parallel audio+face extraction)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.fusion_mode:
        config.setdefault("fusion", {})["mode"] = args.fusion_mode

    print(f"\nAnalysing: {args.video}")
    print("=" * 60)

    if args.async_mode:
        result = asyncio.run(detect_video_async(args.video, config, args.skip))
    else:
        result = detect_video(args.video, config, args.skip)

    print("\n" + "=" * 60)
    print(result)

    # Exit code: 1 = fake, 0 = real (useful for scripting)
    sys.exit(1 if result.is_fake else 0)


if __name__ == "__main__":
    main()
