#!/usr/bin/env python3
"""
Aggregate experiment results into a comparison table.

Reads every JSON file under exp/results/ and prints an ASCII / Markdown table.

Usage
-----
python exp/report.py                     # default: exp/results/
python exp/report.py --results_dir /path/to/results --fmt markdown
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


DETECTOR_ORDER = ["xception", "tscan", "sync"]
METRIC_COLS = ["auc", "acc", "eer", "ap"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_results(results_dir: str) -> List[dict]:
    rdir = Path(results_dir)
    results = []
    for path in sorted(rdir.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            data.setdefault("_file", path.name)
            results.append(data)
        except Exception as e:
            print(f"[WARN] Could not read {path}: {e}", file=sys.stderr)
    return results


def _sort_key(r: dict) -> tuple:
    det = r.get("detector", "")
    ds = r.get("dataset", "")
    det_idx = DETECTOR_ORDER.index(det) if det in DETECTOR_ORDER else 99
    return (ds, det_idx, det)


def _fmt_float(v, width: int = 6) -> str:
    if v is None:
        return " " * width + "-"
    return f"{float(v):{width}.4f}"


# ---------------------------------------------------------------------------
# Table renderers
# ---------------------------------------------------------------------------

def _render_ascii(results: List[dict]) -> str:
    col_det = 12
    col_ds = 32
    col_m = 8
    header = (
        f"  {'DETECTOR':<{col_det}} {'DATASET':<{col_ds}}"
        + "".join(f" {m.upper():>{col_m}}" for m in METRIC_COLS)
        + f"  {'N_REAL':>7} {'N_FAKE':>7}"
    )
    sep = "-" * len(header)
    lines = ["=" * len(header), header, sep]

    prev_ds = None
    for r in results:
        ds = r.get("dataset", "?")
        if ds != prev_ds and prev_ds is not None:
            lines.append(sep)
        prev_ds = ds
        row = (
            f"  {r.get('detector','?'):<{col_det}} {ds:<{col_ds}}"
            + "".join(f" {_fmt_float(r.get(m))}" for m in METRIC_COLS)
            + f"  {r.get('n_real', 0):>7} {r.get('n_fake', 0):>7}"
        )
        lines.append(row)

    lines.append("=" * len(header))
    return "\n".join(lines)


def _render_markdown(results: List[dict]) -> str:
    header = (
        "| Detector | Dataset | AUC | ACC | EER | AP | N_real | N_fake |"
    )
    sep = "|" + "|".join(["-" * 10] * 8) + "|"
    lines = [header, sep]
    for r in results:
        row = (
            f"| {r.get('detector','?')} "
            f"| {r.get('dataset','?')} "
            + "".join(
                f"| {_fmt_float(r.get(m)).strip()} " for m in METRIC_COLS
            )
            + f"| {r.get('n_real',0)} | {r.get('n_fake',0)} |"
        )
        lines.append(row)
    return "\n".join(lines)


def _render_csv(results: List[dict]) -> str:
    header = "detector,dataset," + ",".join(METRIC_COLS) + ",n_real,n_fake"
    lines = [header]
    for r in results:
        vals = [r.get("detector", ""), r.get("dataset", "")]
        vals += [str(r.get(m, "")) for m in METRIC_COLS]
        vals += [str(r.get("n_real", 0)), str(r.get("n_fake", 0))]
        lines.append(",".join(vals))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-dataset breakdown
# ---------------------------------------------------------------------------

def _per_dataset_summary(results: List[dict]) -> str:
    """Group by dataset; show best detector per metric."""
    by_ds: Dict[str, List[dict]] = {}
    for r in results:
        ds = r.get("dataset", "?")
        by_ds.setdefault(ds, []).append(r)

    lines = ["\n── Best detector per dataset ──"]
    for ds, rows in sorted(by_ds.items()):
        lines.append(f"\n  {ds}:")
        for metric in METRIC_COLS:
            # For EER, lower is better; for others, higher is better
            reverse = metric != "eer"
            ranked = sorted(
                [r for r in rows if r.get(metric) is not None],
                key=lambda r: float(r[metric]),
                reverse=reverse,
            )
            if ranked:
                best = ranked[0]
                lines.append(
                    f"    {metric.upper():>4}: {best.get('detector','?'):<12} "
                    f"{float(best[metric]):.4f}"
                )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarise ADB experiment results")
    p.add_argument(
        "--results_dir",
        default=str(Path(__file__).parent / "results"),
        help="Directory containing result JSONs",
    )
    p.add_argument(
        "--fmt",
        choices=["ascii", "markdown", "csv"],
        default="ascii",
        help="Output format",
    )
    p.add_argument("--out", default="", help="Write output to file instead of stdout")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    results = _load_results(args.results_dir)
    if not results:
        print(f"No JSON results found in {args.results_dir}.")
        print("Run  python exp/run_exp.py --help  to start experiments.")
        return

    results.sort(key=_sort_key)

    if args.fmt == "ascii":
        table = _render_ascii(results)
    elif args.fmt == "markdown":
        table = _render_markdown(results)
    else:
        table = _render_csv(results)

    summary = _per_dataset_summary(results)
    output = table + "\n" + summary + "\n"

    if args.out:
        Path(args.out).write_text(output)
        print(f"Report written to {args.out}")
    else:
        print(output)


if __name__ == "__main__":
    main()
