"""
Elder-scam call demo — rich terminal UI.

Simulates a video call scenario where an elderly user receives a suspicious
call. Analyses the video in 5-second windows and alerts the user if deepfake
is detected in two or more consecutive windows.

Usage:
    python demo/scam_call_demo.py --video suspicious_call.mp4 --caller "王小明（兒子）"

Requires: rich >= 13.0.0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

PARETO_CSV = ROOT / "eval" / "pareto_configs.csv"


# ── rich UI helpers ───────────────────────────────────────────────────────────

def _get_console():
    from rich.console import Console
    return Console()


def _risk_color(decision: str) -> str:
    return "bold red" if decision == "FAKE" else "green"


def _risk_emoji(decision: str) -> str:
    return "[bold red]⚠ SUSPICIOUS[/bold red]" if decision == "FAKE" else "[green]✓ OK[/green]"


# ── Pareto config loading (mirrors run_pipeline.py) ───────────────────────────

def _load_best_serial() -> Optional[dict]:
    if not PARETO_CSV.exists():
        return None
    with open(PARETO_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    serial = [r for r in rows if r.get("mode") == "serial"]
    acceptable = [r for r in serial if float(r.get("system_FAR", 1)) <= 0.05]
    if not acceptable:
        return None
    return min(acceptable, key=lambda r: float(r.get("total_time_ms", 9999)))


# ── window utilities ──────────────────────────────────────────────────────────

def _split_windows(track, fps: float, window_sec: float = 5.0):
    from preprocessing.face_extractor import FaceTrack
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


# ── main demo ─────────────────────────────────────────────────────────────────

def run_demo(video_path: Path, caller_name: str, config_path: Path) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box

    console = _get_console()

    import yaml
    cfg_yaml: dict = {}
    if config_path.exists():
        with open(config_path) as f:
            cfg_yaml = yaml.safe_load(f) or {}

    # ── Incoming call banner ──────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        f"[bold cyan]📞  來電：{caller_name}[/bold cyan]\n"
        f"[dim]Anti-Deepfake-Box 即時偵測已啟動[/dim]",
        title="[bold]⚠ 疑似詐騙電話偵測[/bold]",
        border_style="cyan",
        padding=(1, 4),
    ))
    console.print()
    time.sleep(0.5)

    # ── Load models ───────────────────────────────────────────────────────────
    console.print("[dim]正在載入偵測模型...[/dim]")
    from detectors.visual_detector import VisualDetector
    from detectors.rppg_detector import RPPGDetector
    from detectors.sync_detector import SyncDetector

    det_cfg = cfg_yaml.get("detectors", {})
    visual_det = VisualDetector(det_cfg.get("visual", {}))
    visual_det.load()
    rppg_det = RPPGDetector(det_cfg.get("rppg", {}))
    rppg_det.load()
    sync_det = SyncDetector(det_cfg.get("sync", {}))
    sync_det.load()

    # ── Load serial config ────────────────────────────────────────────────────
    pareto_cfg = _load_best_serial()
    use_fallback = pareto_cfg is None
    if pareto_cfg:
        stage_order = pareto_cfg["stage_order"].split(",")
        H_vals = [float(h) for h in pareto_cfg["H"].split(",")] if pareto_cfg.get("H") else []
        L_vals = [float(l) for l in pareto_cfg["L"].split(",")] if pareto_cfg.get("L") else []
        F = float(pareto_cfg.get("final_threshold", 0.5))
        serial_config = {
            "config_id": pareto_cfg["config_id"],
            "stage_order": stage_order,
            "high_thresholds": H_vals,
            "low_thresholds": L_vals,
            "final_threshold": F,
        }
    else:
        console.print("[dim]未找到 Pareto 配置，使用加權融合 fallback[/dim]")

    # ── Extract faces ─────────────────────────────────────────────────────────
    from preprocessing.face_extractor import UnifiedFaceExtractor
    console.print(f"[dim]提取人臉軌跡：{video_path.name}[/dim]")
    extractor = UnifiedFaceExtractor(cfg_yaml.get("preprocessing", {}))
    track = extractor.extract(video_path)
    fps = track.fps
    console.print(f"[dim]人臉軌跡：{len(track.aligned_frames)} 幀 @ {fps:.1f} fps[/dim]\n")

    # ── Per-window table ──────────────────────────────────────────────────────
    table = Table(title="即時偵測視窗", box=box.ROUNDED, show_lines=False)
    table.add_column("視窗", justify="center", style="dim", width=6)
    table.add_column("視覺", justify="right", width=8)
    table.add_column("rPPG", justify="right", width=8)
    table.add_column("同步", justify="right", width=8)
    table.add_column("使用關卡", justify="center", width=8)
    table.add_column("判斷", justify="center", width=12)
    table.add_column("耗時", justify="right", width=8)

    window_log: List[dict] = []
    consecutive_fake = 0
    alert_raised = False

    console.print(table)  # Print header immediately; we'll reprint rows live

    for win_idx, n_wins, sub_track in _split_windows(track, fps):
        t0 = time.perf_counter()

        scores: Dict[str, float] = {}
        try:
            s = visual_det.predict(sub_track)
            if s is not None:
                scores["visual_v2"] = float(s)
        except Exception:
            pass
        try:
            s = rppg_det.predict(sub_track)
            if s is not None:
                scores["rppg_v2"] = float(s)
        except Exception:
            pass
        try:
            s = sync_det.predict(sub_track)
            if s is not None:
                scores["sync_v1"] = float(s)
        except Exception:
            pass

        dt_ms = (time.perf_counter() - t0) * 1000

        if use_fallback:
            from fusion.weighted_ensemble import WeightedEnsemble
            ens = WeightedEnsemble(cfg_yaml.get("fusion", {}))
            result = ens.fuse(scores)
            decision = result.prediction
            stages_used = len(scores)
        else:
            from fusion.serial_fusion import serial_decision
            res = serial_decision(scores, serial_config)
            decision = res.decision
            stages_used = res.stages_used

        vis_str  = f"{scores.get('visual_v2', '--'):.3f}" if "visual_v2" in scores else "--"
        rppg_str = f"{scores.get('rppg_v2', '--'):.3f}"  if "rppg_v2"  in scores else "--"
        sync_str = f"{scores.get('sync_v1', '--'):.3f}"  if "sync_v1"  in scores else "--"
        color = "bold red" if decision == "FAKE" else "green"
        label = "[bold red]⚠ 可疑[/bold red]" if decision == "FAKE" else "[green]✓ 正常[/green]"

        console.print(
            f"  [{win_idx+1:02d}/{n_wins}] "
            f"visual:{vis_str}  rppg:{rppg_str}  sync:{sync_str}  "
            f"stages:{stages_used}  {label}  ({dt_ms:.0f}ms)"
        )

        window_log.append({
            "window": win_idx + 1,
            "decision": decision,
            "scores": dict(scores),
            "stages_used": stages_used,
            "time_ms": dt_ms,
        })

        if decision == "FAKE":
            consecutive_fake += 1
        else:
            consecutive_fake = 0

        # Alert after 2 consecutive FAKE windows
        if consecutive_fake >= 2 and not alert_raised:
            alert_raised = True
            console.print()
            console.print(Panel(
                "[bold white on red]  ⛔  偵測到連續多個可疑片段！  ⛔  [/bold white on red]\n\n"
                "[bold red]此通話極可能是 DEEPFAKE 詐騙！[/bold red]\n"
                f"來電者聲稱是「{caller_name}」，但影像特徵顯示異常。\n\n"
                "[yellow]建議：[/yellow]\n"
                "  1. 立即掛斷電話\n"
                "  2. 直接撥打對方真實號碼確認身份\n"
                "  3. 不要轉帳或提供個人資訊",
                title="[bold red]⚠ 詐騙警告[/bold red]",
                border_style="red",
                padding=(1, 4),
            ))
            console.print()

    # ── Summary ───────────────────────────────────────────────────────────────
    fake_count = sum(1 for w in window_log if w["decision"] == "FAKE")
    fake_ratio = fake_count / max(len(window_log), 1)
    avg_stages = sum(w["stages_used"] for w in window_log) / max(len(window_log), 1)

    if fake_ratio >= 0.6:
        risk = "[bold red]高風險 (HIGH)[/bold red]"
    elif fake_ratio >= 0.3:
        risk = "[yellow]中風險 (MEDIUM)[/yellow]"
    else:
        risk = "[green]低風險 (LOW)[/green]"

    console.print()
    console.print(Panel(
        f"[bold]通話分析摘要[/bold]\n\n"
        f"  總視窗數  : {len(window_log)}\n"
        f"  FAKE 比例 : {fake_ratio:.0%}  ({fake_count}/{len(window_log)} 視窗)\n"
        f"  平均關卡  : {avg_stages:.1f} / {len(scores)} 個模組\n"
        f"  風險等級  : {risk}",
        title="[bold]📊 通話結束[/bold]",
        border_style="cyan",
        padding=(1, 4),
    ))

    # Save log
    from pathlib import Path as P
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    log_path = results_dir / f"scam_call_{video_path.stem}.json"
    with open(log_path, "w") as f:
        json.dump({
            "caller": caller_name,
            "video": str(video_path),
            "windows": window_log,
            "fake_ratio": fake_ratio,
            "avg_stages": avg_stages,
            "alert_raised": alert_raised,
        }, f, indent=2, ensure_ascii=False)
    console.print(f"\n[dim]通話記錄已儲存 → {log_path}[/dim]\n")


def main():
    ap = argparse.ArgumentParser(
        description="Elder scam-call deepfake demo (rich UI)"
    )
    ap.add_argument("--video",  type=Path, required=True, help="Suspicious call video (.mp4)")
    ap.add_argument("--caller", default="未知來電者", help="Caller display name")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    args = ap.parse_args()

    if not args.video.exists():
        print(f"Error: video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    run_demo(args.video, args.caller, args.config)


if __name__ == "__main__":
    main()
