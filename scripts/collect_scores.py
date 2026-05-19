#!/usr/bin/env python3
"""
Unified weak-classifier score collection for Anti-Deepfake-Box.

Runs selected detectors (visual / rPPG / sync) on a video directory in a
single pass — one face extraction shared across all modalities (SSOT) — and
writes three separate CSVs in the 12-column GP-solver data contract format.

Output CSVs are named {modality}_{detector_key}_scores.csv so multiple runs
with different detector choices can coexist in the same output_dir.
cascade_selection.py loads all *_scores.csv files from a directory and selects
the best per-modality detector automatically.

Output format (matches Week 11 GP data contract / fusion_solver_prod_v1.ipynb):
    sample_id, dataset, label, detector_name, modality, fake_score,
    score_type, inference_time_ms, window_start_sec, window_end_sec,
    status, error_message

Usage
-----
# Default (Xception + POS + SyncNet):
python scripts/collect_scores.py \\
    --input_dir /data/FakeAVCeleb/test \\
    --label_csv /data/FakeAVCeleb/labels.csv \\
    --output_dir results/ \\
    --dataset FakeAVCeleb \\
    --mode forensic

# Use UCF visual detector (requires DeepfakeBench on DFB_PATH):
python scripts/collect_scores.py \\
    --input_dir /data/FakeAVCeleb/test \\
    --output_dir results/ \\
    --visual_detector ucf \\
    --dfb_pretrained /ckpts/ucf_ff.pth

# Skip sync (for FF++ which has no audio):
python scripts/collect_scores.py \\
    --input_dir /data/FF++/test \\
    --output_dir results/ff \\
    --dataset FF++ \\
    --skip sync

# CPU-only with SBI:
python scripts/collect_scores.py \\
    --input_dir /data/test \\
    --output_dir results/ \\
    --visual_detector sbi \\
    --device cpu

Notes
-----
* Failed samples are NEVER dropped. status="failed" rows are written with
  fake_score="" so the GP data_validator_v1 can convert them to NaN
  and apply its fallback mechanism without sample_id misalignment.
* Videos with no audio track: sync row gets status="failed",
  error_message="no_audio_track".
* Leading silence skip (forensic mode): 80 ms, matches DFB-MM §B.1.
* DFB-backed detectors require: export DFB_PATH=/path/to/DeepfakeBench
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
from detectors.registry import VISUAL_REGISTRY, RPPG_REGISTRY, SYNC_REGISTRY, DEFAULTS, build_detector


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
    visual_detector: str = "xception",
    rppg_detector: str = "pos",
    sync_detector: str = "syncnet",
    dfb_pretrained: str = "",
    visual_pretrained: str = "",
) -> Dict[str, List[Dict]]:
    """
    Run selected detectors on every video using a single face extraction pass.

    Returns dict keyed by "{modality}_{detector_key}" (e.g. "visual_xception").
    CSV filename: {key}_scores.csv — multiple runs coexist in the same output_dir.
    Failed rows are included with status="failed" — never dropped.
    """
    label_map = label_map or {}
    skip = skip or []

    # Per-modality detector configs
    vis_cfg = {
        "device": device,
        "pretrained": visual_pretrained,
        "dfb_pretrained": dfb_pretrained,
    }
    rppg_cfg = {"device": "cpu"}
    sync_cfg = {
        "device": device,
        "syncnet_path": syncnet_path,
        "whisper_model": whisper_model,
        "whisper_device": "cpu",
    }

    # Build active detectors via registry
    # active_det: modality → detector instance
    # active_meta: modality → {detector_name, modality, score_type, ...}
    # active_key: modality → detector key string
    active_det: Dict[str, object] = {}
    active_meta: Dict[str, dict] = {}
    active_key: Dict[str, str] = {}

    if "visual" not in skip:
        det, meta = build_detector("visual", visual_detector, vis_cfg)
        active_det["visual"] = det
        active_meta["visual"] = meta
        active_key["visual"] = visual_detector
        print(f"[visual]  {meta['detector_name']}  ({meta['status']})")

    if "rppg" not in skip:
        det, meta = build_detector("rppg", rppg_detector, rppg_cfg)
        active_det["rppg"] = det
        active_meta["rppg"] = meta
        active_key["rppg"] = rppg_detector
        print(f"[rppg]    {meta['detector_name']}  ({meta['status']})")

    if "sync" not in skip:
        det, meta = build_detector("sync", sync_detector, sync_cfg)
        active_det["sync"] = det
        active_meta["sync"] = meta
        active_key["sync"] = sync_detector
        print(f"[sync]    {meta['detector_name']}  ({meta['status']})")

    print()

    # Output rows keyed by "{modality}_{key}" for distinct CSV filenames
    rows: Dict[str, List[Dict]] = {
        f"{active_meta[m]['modality']}_{active_key[m]}": []
        for m in active_det
    }
    # Map from modality → row_key for inner loop
    row_key_for = {m: f"{active_meta[m]['modality']}_{active_key[m]}" for m in active_det}

    # Build extractors
    audio_ext = AudioExtractor({
        "sample_rate": 16000,
        "channels": 1,
        "skip_leading_ms": skip_leading_ms,
    })
    face_ext = UnifiedFaceExtractor({"device": device, "use_face_cache": True})

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
        if "sync" in active_det:
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
        for modality, det in active_det.items():
            t0 = time.time()
            score: Optional[float] = None
            status = "ok"
            err = ""

            try:
                if face_track is None:
                    raise RuntimeError(face_err or "face_not_detected")
                if modality == "sync":
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
            meta = active_meta[modality]
            rows[row_key_for[modality]].append({
                "sample_id":         vp.stem,
                "dataset":           dataset_name,
                "label":             label_map.get(vp.stem, ""),
                "detector_name":     meta["detector_name"],
                "modality":          meta["modality"],
                "score_type":        meta["score_type"],
                "fake_score":        f"{score:.6f}" if score is not None else "",
                "inference_time_ms": ms,
                "window_start_sec":  "N/A",
                "window_end_sec":    "N/A",
                "status":            status,
                "error_message":     err,
            })

            tag = f"{score:.4f}" if score is not None else f"FAILED({err})"
            det_name = meta["detector_name"]
            print(f"  [{modality:6s}/{det_name:12s}] {tag}  ({ms:.0f} ms)")

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

def write_csvs(rows_by_key: Dict[str, List[Dict]], output_dir: str) -> None:
    """Write one CSV per detector run. Filename: {modality}_{detector_key}_scores.csv."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for key, rows in rows_by_key.items():
        out = Path(output_dir) / f"{key}_scores.csv"
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        ok = sum(1 for r in rows if r["status"] == "ok")
        print(f"  {key:30s} → {out.name}  ({ok}/{len(rows)} ok)")

    print()
    for key, rows in rows_by_key.items():
        scored = [float(r["fake_score"]) for r in rows if r["fake_score"]]
        if scored:
            print(f"  [{key}] range [{min(scored):.3f}, {max(scored):.3f}]"
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
    p.add_argument("--syncnet_path", default="",          help="Optional SyncNet checkpoint path")
    p.add_argument("--whisper_model", default="tiny",
                   choices=["tiny", "base", "small", "medium", "large"])
    p.add_argument("--mode", choices=["forensic", "realtime"], default="realtime",
                   help=(
                       "forensic  — skip 80ms leading silence, whisper=small\n"
                       "realtime  — no silence skip, whisper=tiny [default]"
                   ))
    p.add_argument("--skip", nargs="+", choices=["visual", "rppg", "sync"],
                   default=[], help="Skip specified modalities")

    # ── Detector selection ────────────────────────────────────────────
    p.add_argument(
        "--visual_detector",
        default=DEFAULTS["visual"],
        choices=sorted(VISUAL_REGISTRY),
        help=(
            "Visual detector algorithm (default: xception).\n"
            "Tested: xception, ucf, sbi, f3net, spsl, srm\n"
            "Planned: efficientnet_b4, facexray, lsda\n"
            "DFB-backed detectors require: export DFB_PATH=/path/to/DeepfakeBench"
        ),
    )
    p.add_argument(
        "--rppg_detector",
        default=DEFAULTS["rppg"],
        choices=sorted(RPPG_REGISTRY),
        help="rPPG detector algorithm (default: pos). Backup: tscan (rppg_toolbox required)",
    )
    p.add_argument(
        "--sync_detector",
        default=DEFAULTS["sync"],
        choices=sorted(SYNC_REGISTRY),
        help="AV-sync detector algorithm (default: syncnet). Others are planned stubs.",
    )
    p.add_argument(
        "--dfb_pretrained", default="",
        help="Path to DFB checkpoint for DFB-backed visual detectors (ucf/sbi/f3net/...)",
    )
    p.add_argument(
        "--visual_pretrained", default="",
        help="Path to Xception checkpoint (used when --visual_detector xception)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    skip_ms = 80 if args.mode == "forensic" else 0
    whisper = args.whisper_model
    if args.mode == "forensic" and whisper == "tiny":
        whisper = "small"

    print(f"[mode={args.mode}]  skip_leading_ms={skip_ms}  whisper={whisper}")
    print(f"[detectors]  visual={args.visual_detector}  rppg={args.rppg_detector}  sync={args.sync_detector}")
    if args.skip:
        print(f"[skip] {args.skip}")
    print()

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
        visual_detector=args.visual_detector,
        rppg_detector=args.rppg_detector,
        sync_detector=args.sync_detector,
        dfb_pretrained=args.dfb_pretrained,
        visual_pretrained=args.visual_pretrained,
    )

    print(f"\nWrote {sum(len(v) for v in rows.values())} rows to {args.output_dir}/")
    write_csvs(rows, args.output_dir)


if __name__ == "__main__":
    main()
