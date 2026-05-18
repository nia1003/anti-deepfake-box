#!/usr/bin/env python3
"""
Standalone sync-score CSV bridge for Anti-Deepfake-Box.

Runs SyncDetector (audio-visual lip-sync) on a directory of videos and
writes a CSV with fake_score + timing per video — no DeepfakeBench needed.

Output format matches 曉蓮's GP-solver input:
    video_id, fake_score, label, inference_ms

Usage
-----
# Score all .mp4 files in a directory:
python scripts/sync_score_csv.py \\
    --input_dir /data/FakeAVCeleb/test \\
    --output results/sync_scores.csv

# With ground-truth labels (CSV: video_id,label where 0=real,1=fake):
python scripts/sync_score_csv.py \\
    --input_dir /data/FakeAVCeleb/test \\
    --label_csv /data/FakeAVCeleb/labels.csv \\
    --output results/sync_scores.csv

# Process a specific list of video paths (one per line):
python scripts/sync_score_csv.py \\
    --video_list my_videos.txt \\
    --output results/sync_scores.csv \\
    --device cpu

Notes
-----
* Videos without an audio track are recorded with fake_score='' and skipped
  in downstream GP-solver analysis (expected for FF++ which has no audio).
* SyncNet weights are optional; if unavailable the detector falls back to a
  motion heuristic (lower accuracy but always produces a score).
* Set PYTHONPATH to include anti-deepfake-box root if importing from outside:
    export PYTHONPATH=/path/to/anti-deepfake-box:$PYTHONPATH
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing.audio_extractor import AudioExtractor
from preprocessing.face_extractor import UnifiedFaceExtractor
from detectors.sync_detector import SyncDetector


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _load_label_map(label_csv: str) -> Dict[str, str]:
    """
    Read a CSV with columns video_id,label (0=real, 1=fake).
    Returns {video_id: label_str}.
    """
    label_map: Dict[str, str] = {}
    with open(label_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("video_id", row.get("filename", "")).strip()
            # Strip extension so we can match by stem
            vid = Path(vid).stem if vid else ""
            lbl = row.get("label", "").strip()
            if vid:
                label_map[vid] = lbl
    return label_map


def _collect_videos(input_dir: Optional[str], video_list: Optional[str]) -> List[Path]:
    """Collect .mp4 / .avi / .mov video paths from directory or list file."""
    if video_list:
        paths = []
        with open(video_list) as f:
            for line in f:
                p = Path(line.strip())
                if p.exists():
                    paths.append(p)
        return sorted(paths)
    if input_dir:
        exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        return sorted(p for p in Path(input_dir).rglob("*") if p.suffix.lower() in exts)
    return []


# ------------------------------------------------------------------ #
#  Core scoring loop                                                   #
# ------------------------------------------------------------------ #

def score_videos(
    videos: List[Path],
    device: str = "cuda",
    syncnet_path: str = "",
    whisper_model: str = "tiny",
    label_map: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """
    Run SyncDetector on each video and return rows for CSV output.

    Returns list of dicts: {video_id, fake_score, label, inference_ms}
    """
    label_map = label_map or {}

    detector_cfg = {
        "device": device,
        "syncnet_path": syncnet_path,
        "whisper_model": whisper_model,
        "whisper_device": "cpu",  # keep Whisper on CPU to avoid VRAM contention
    }
    face_cfg = {"device": device, "use_face_cache": True}

    audio_ext = AudioExtractor({"sample_rate": 16000, "channels": 1})
    face_ext = UnifiedFaceExtractor(face_cfg)
    detector = SyncDetector(detector_cfg)

    rows = []
    total = len(videos)

    for idx, vp in enumerate(videos, 1):
        print(f"[{idx}/{total}] {vp.name}", end=" ... ", flush=True)
        t0 = time.time()

        wav_path: Optional[str] = None
        face_track = None
        fake_score: Optional[float] = None

        try:
            # Audio extraction (graceful: None if no audio track)
            if audio_ext.has_audio(str(vp)):
                wav_path = audio_ext.extract_to_temp(str(vp))

            # Face track extraction
            face_track = face_ext.extract(str(vp))

            # Sync detection (returns None if no audio or no face)
            if face_track is not None and wav_path is not None:
                fake_score = detector.detect(face_track, wav_path)

        except Exception as exc:
            print(f"ERROR: {exc}")
        finally:
            # Clean up temp WAV
            if wav_path and Path(wav_path).exists():
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

        ms = round((time.time() - t0) * 1000, 1)
        score_str = f"{fake_score:.6f}" if fake_score is not None else ""
        status = score_str if score_str else "no_audio"
        print(f"{status}  ({ms} ms)")

        rows.append({
            "video_id": vp.stem,
            "fake_score": score_str,
            "label": label_map.get(vp.stem, ""),
            "inference_ms": ms,
        })

    return rows


# ------------------------------------------------------------------ #
#  Output                                                              #
# ------------------------------------------------------------------ #

def write_csv(rows: List[Dict], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["video_id", "fake_score", "label", "inference_ms"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows → {output_path}")

    # Quick summary
    scored = [r for r in rows if r["fake_score"]]
    print(f"  Scored:    {len(scored)}/{len(rows)} videos (rest had no audio)")
    if scored:
        scores = [float(r["fake_score"]) for r in scored]
        print(f"  Score range: [{min(scores):.3f}, {max(scores):.3f}]  mean={sum(scores)/len(scores):.3f}")


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute SyncDetector fake scores for a batch of videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_dir", help="Directory containing video files (recursive)")
    src.add_argument("--video_list", help="Text file with one video path per line")

    p.add_argument("--output", default="sync_scores.csv", help="Output CSV path")
    p.add_argument("--label_csv", default="", help="Optional CSV with video_id,label columns")
    p.add_argument("--device", default="cuda", help="Torch device: cuda or cpu")
    p.add_argument("--syncnet_path", default="", help="Optional SyncNet checkpoint path")
    p.add_argument("--whisper_model", default="tiny",
                   choices=["tiny", "base", "small", "medium", "large"],
                   help="Whisper model size (tiny is fastest; default: tiny)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    videos = _collect_videos(args.input_dir, args.video_list)
    if not videos:
        print("No video files found. Check --input_dir or --video_list.")
        sys.exit(1)

    print(f"Found {len(videos)} video(s).")

    label_map: Dict[str, str] = {}
    if args.label_csv:
        label_map = _load_label_map(args.label_csv)
        print(f"Loaded {len(label_map)} labels from {args.label_csv}")

    rows = score_videos(
        videos=videos,
        device=args.device,
        syncnet_path=args.syncnet_path,
        whisper_model=args.whisper_model,
        label_map=label_map,
    )
    write_csv(rows, args.output)


if __name__ == "__main__":
    main()
