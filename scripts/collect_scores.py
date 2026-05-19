#!/usr/bin/env python3
"""
Unified weak-classifier score collection for Anti-Deepfake-Box.

Runs all three detectors (visual / rPPG / sync) on a video directory in a
single pass — one face extraction shared across all modalities (SSOT) — and
writes three separate CSVs in the 12-column GP-solver data contract format.

Output format (matches Week 11 GP data contract / fusion_solver_prod_v1.ipynb):
    sample_id, dataset, label, detector_name, modality, fake_score,
    score_type, inference_time_ms, window_start_sec, window_end_sec,
    status, error_message

GP solver loads all three files via:
    load_real_data(['results/visual_scores.csv',
                    'results/rppg_scores.csv',
                    'results/sync_scores.csv'])
and aligns rows by sample_id.

Usage
-----
# All three modalities:
python scripts/collect_scores.py \\
    --input_dir /data/FakeAVCeleb/test \\
    --label_csv /data/FakeAVCeleb/labels.csv \\
    --output_dir results/ \\
    --dataset FakeAVCeleb \\
    --mode forensic

# Skip sync (for FF++ which has no audio):
python scripts/collect_scores.py \\
    --input_dir /data/FF++/test \\
    --output_dir results/ff \\
    --dataset FF++ \\
    --skip sync

# CPU-only:
python scripts/collect_scores.py \\
    --input_dir /data/test \\
    --output_dir results/ \\
    --device cpu

Notes
-----
* Failed samples are NEVER dropped. status="failed" rows are written with
  fake_score="" so the GP data_validator_v1 can convert them to NaN
  and apply its fallback mechanism without sample_id misalignment.
* Videos with no audio track: sync row gets status="failed",
  error_message="no_audio_track".
* Leading silence skip (forensic mode): 80 ms, matches DFB-MM §B.1.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing.audio_extractor import AudioExtractor
from preprocessing.face_extractor import UnifiedFaceExtractor
from detectors.visual_detector import VisualDetector
from detectors.rppg_detector import RPPGDetector
from detectors.sync_detector import SyncDetector


# ------------------------------------------------------------------ #
#  GP data contract — 12 standard columns                             #
# ------------------------------------------------------------------ #

FIELDNAMES = [
    "sample_id",
    "dataset",
    "label",
    "detector_name",
    "modality",
    "fake_score",
    "score_type",
    "inference_time_ms",
    "window_start_sec",
    "window_end_sec",
    "status",
    "error_message",
]

DETECTOR_META: Dict[str, Dict[str, str]] = {
    "visual": {
        "detector_name": "Xception",
        "modality":      "visual",
        "score_type":    "probability",
    },
    "rppg": {
        "detector_name": "POS",
        "modality":      "rppg",
        "score_type":    "snr",
    },
    "sync": {
        "detector_name": "SyncNet",
        "modality":      "av_sync",
        "score_type":    "sync_error",
    },
}


# ------------------------------------------------------------------ #
#  Helpers (inline — mirrors sync_score_csv.py helpers)               #
# ------------------------------------------------------------------ #

def _load_label_map(label_csv: str) -> Dict[str, str]:
    label_map: Dict[str, str] = {}
    with open(label_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("video_id", row.get("filename", row.get("sample_id", ""))).strip()
            vid = Path(vid).stem if vid else ""
            lbl = row.get("label", "").strip()
            if vid:
                label_map[vid] = lbl
    return label_map


def _collect_videos(input_dir: Optional[str], video_list: Optional[str]) -> List[Path]:
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

def score_all_detectors(
    videos: List[Path],
    device: str = "cuda",
    syncnet_path: str = "",
    whisper_model: str = "tiny",
    label_map: Optional[Dict[str, str]] = None,
    skip_leading_ms: int = 0,
    dataset_name: str = "",
    skip: Optional[List[str]] = None,
) -> Dict[str, List[Dict]]:
    """
    Run all three detectors on every video using a single face extraction pass.

    Returns dict mapping modality name → list of row dicts (12 columns each).
    Failed rows are included with status="failed" — never dropped.
    """
    label_map = label_map or {}
    skip = skip or []

    # Build extractors
    audio_ext = AudioExtractor({
        "sample_rate": 16000,
        "channels": 1,
        "skip_leading_ms": skip_leading_ms,
    })
    face_ext = UnifiedFaceExtractor({"device": device, "use_face_cache": True})

    # Build active detectors
    active: Dict[str, object] = {}
    if "visual" not in skip:
        active["visual"] = VisualDetector({"device": device})
    if "rppg" not in skip:
        active["rppg"] = RPPGDetector({"device": "cpu"})
    if "sync" not in skip:
        active["sync"] = SyncDetector({
            "device": device,
            "syncnet_path": syncnet_path,
            "whisper_model": whisper_model,
            "whisper_device": "cpu",
        })

    rows: Dict[str, List[Dict]] = {name: [] for name in active}
    total = len(videos)

    for idx, vp in enumerate(videos, 1):
        print(f"[{idx}/{total}] {vp.name}", flush=True)

        # ── SSOT face extraction ──────────────────────────────────────
        face_track = None
        face_err = ""
        try:
            face_track = face_ext.extract(str(vp))
            if face_track is None:
                face_err = "face_not_detected"
        except Exception as exc:
            face_err = f"face_extract_error: {str(exc)[:100]}"

        # ── Audio extraction (shared by sync) ─────────────────────────
        wav_path: Optional[str] = None
        audio_err = ""
        if "sync" in active:
            try:
                if audio_ext.has_audio(str(vp)):
                    wav_path = audio_ext.extract_to_temp(str(vp))
                    if wav_path is None:
                        audio_err = "audio_decode_failed"
                else:
                    audio_err = "no_audio_track"
            except Exception as exc:
                audio_err = f"audio_extract_error: {str(exc)[:100]}"

        # ── Per-detector inference ────────────────────────────────────
        for name, det in active.items():
            t0 = time.time()
            score: Optional[float] = None
            status = "ok"
            err = ""

            try:
                if face_track is None:
                    raise RuntimeError(face_err or "face_not_detected")
                if name == "sync":
                    if wav_path is None:
                        raise RuntimeError(audio_err or "no_audio_track")
                    score = det.detect(face_track, wav_path)
                else:
                    score = det.detect(face_track, None)
                if score is None:
                    status, err = "failed", "detector_returned_none"
            except Exception as exc:
                status, err = "failed", str(exc)[:120]

            ms = round((time.time() - t0) * 1000, 1)
            rows[name].append({
                "sample_id":        vp.stem,
                "dataset":          dataset_name,
                "label":            label_map.get(vp.stem, ""),
                **DETECTOR_META[name],
                "fake_score":       f"{score:.6f}" if score is not None else "",
                "inference_time_ms": ms,
                "window_start_sec": "N/A",
                "window_end_sec":   "N/A",
                "status":           status,
                "error_message":    err,
            })

            tag = f"{score:.4f}" if score is not None else f"FAILED({err})"
            print(f"  [{name:6s}] {tag}  ({ms:.0f} ms)")

        # ── Cleanup temp audio ────────────────────────────────────────
        if wav_path and Path(wav_path).exists():
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    return rows


# ------------------------------------------------------------------ #
#  Output                                                              #
# ------------------------------------------------------------------ #

def write_csvs(rows_by_detector: Dict[str, List[Dict]], output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for name, rows in rows_by_detector.items():
        out = Path(output_dir) / f"{name}_scores.csv"
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        ok = sum(1 for r in rows if r["status"] == "ok")
        print(f"  {name:6s} → {out}  ({ok}/{len(rows)} ok)")

    # Quick score summary
    print()
    for name, rows in rows_by_detector.items():
        scored = [float(r["fake_score"]) for r in rows if r["fake_score"]]
        if scored:
            print(f"  [{name}] range [{min(scored):.3f}, {max(scored):.3f}]"
                  f"  mean={sum(scored)/len(scored):.3f}  n={len(scored)}")


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect per-modality fake scores for GP solver input.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_dir",  help="Directory of video files (recursive)")
    src.add_argument("--video_list", help="Text file with one video path per line")

    p.add_argument("--output_dir",   default="results",   help="Output directory for CSV files")
    p.add_argument("--label_csv",    default="",          help="Optional CSV with sample_id/video_id,label")
    p.add_argument("--dataset",      default="",          help="Dataset name written to CSV (e.g. FakeAVCeleb)")
    p.add_argument("--device",       default="cuda",      help="Torch device: cuda or cpu")
    p.add_argument("--syncnet_path", default="",          help="Optional SyncNet checkpoint")
    p.add_argument("--whisper_model", default="tiny",
                   choices=["tiny", "base", "small", "medium", "large"])
    p.add_argument("--mode", choices=["forensic", "realtime"], default="realtime",
                   help=(
                       "forensic  — skip 80ms leading silence, whisper=small\n"
                       "realtime  — no silence skip, whisper=tiny [default]"
                   ))
    p.add_argument("--skip", nargs="+", choices=["visual", "rppg", "sync"],
                   default=[], help="Skip specified modalities")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    skip_ms = 80 if args.mode == "forensic" else 0
    whisper = args.whisper_model
    if args.mode == "forensic" and whisper == "tiny":
        whisper = "small"
    print(f"[mode={args.mode}]  skip_leading_ms={skip_ms}  whisper={whisper}")
    if args.skip:
        print(f"[skip] {args.skip}")

    videos = _collect_videos(args.input_dir, args.video_list)
    if not videos:
        print("No video files found.")
        sys.exit(1)
    print(f"Found {len(videos)} video(s).\n")

    label_map: Dict[str, str] = {}
    if args.label_csv:
        label_map = _load_label_map(args.label_csv)
        print(f"Loaded {len(label_map)} labels from {args.label_csv}")

    rows = score_all_detectors(
        videos=videos,
        device=args.device,
        syncnet_path=args.syncnet_path,
        whisper_model=whisper,
        label_map=label_map,
        skip_leading_ms=skip_ms,
        dataset_name=args.dataset,
        skip=args.skip,
    )

    print(f"\nWrote {sum(len(v) for v in rows.values())} rows to {args.output_dir}/")
    write_csvs(rows, args.output_dir)


if __name__ == "__main__":
    main()
