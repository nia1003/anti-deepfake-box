"""
Detector registry for Anti-Deepfake-Box.

Each entry maps a detector key → {factory, detector_name, modality, score_type, status}.

Status tags:
  "available"     — implemented and tested in ADB
  "dfb_required"  — wraps a DeepfakeBench detector; DFB must be discoverable (DFB_PATH env or sibling dir)
  "rppg_toolbox"  — wraps an rPPG-Toolbox detector; rPPG-Toolbox must be on path
  "planned"       — interface defined; weights/training pending

Usage:
    from detectors.registry import build_detector
    detector, meta = build_detector("visual", "ucf", {"device": "cuda", "dfb_pretrained": "/ckpt.pth"})

CSV column 'detector_name' comes from meta["detector_name"].
CSV filename from collect_scores.py: {meta['modality']}_{key}_scores.csv
"""

from __future__ import annotations

from typing import Any, Dict, Tuple


# ---- Visual (modality = "visual") ----------------------------------------

def _vis_xception(config: dict):
    from .visual_detector import VisualDetector
    return VisualDetector(config)


def _vis_dfb(dfb_name: str):
    def factory(config: dict):
        from .dfb_visual_wrapper import DFBVisualDetector
        return DFBVisualDetector(dfb_name, config)
    return factory


VISUAL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── Tested (score CSVs produced) ──────────────────────────────────────
    "xception": {
        "factory":       _vis_xception,
        "detector_name": "Xception",
        "modality":      "visual",
        "score_type":    "probability",
        "status":        "available",
        "notes":         "ImageNet + FF++ c23 fine-tuned; domain-consistent baseline",
    },
    "ucf": {
        "factory":       _vis_dfb("ucf"),
        "detector_name": "UCF",
        "modality":      "visual",
        "score_type":    "probability",
        "status":        "dfb_required",
        "notes":         "SOTA; strong in-domain and cross-domain robustness",
    },
    "sbi": {
        "factory":       _vis_dfb("sbi"),
        "detector_name": "SBI",
        "modality":      "visual",
        "score_type":    "probability",
        "status":        "dfb_required",
        "notes":         "Best cross-domain generalisation; most robust OOD",
    },
    "f3net": {
        "factory":       _vis_dfb("f3net"),
        "detector_name": "F3Net",
        "modality":      "visual",
        "score_type":    "probability",
        "status":        "dfb_required",
        "notes":         "Very strong in-domain; wide threshold window",
    },
    "spsl": {
        "factory":       _vis_dfb("spsl"),
        "detector_name": "SPSL",
        "modality":      "visual",
        "score_type":    "probability",
        "status":        "dfb_required",
        "notes":         "Tested; systematic high-score bias — monitor threshold carefully",
    },
    "srm": {
        "factory":       _vis_dfb("srm"),
        "detector_name": "SRM",
        "modality":      "visual",
        "score_type":    "probability",
        "status":        "dfb_required",
        "notes":         "Tested; extreme threshold-cliff — usable window very narrow",
    },
    # ── Planned / backup ─────────────────────────────────────────────────
    "efficientnet_b4": {
        "factory":       _vis_dfb("efficientnetb4"),
        "detector_name": "EfficientB4",
        "modality":      "visual",
        "score_type":    "probability",
        "status":        "planned",
        "notes":         "Originally co-listed with Xception; checkpoint pending",
    },
    "facexray": {
        "factory":       _vis_dfb("facexray"),
        "detector_name": "FaceXray",
        "modality":      "visual",
        "score_type":    "probability",
        "status":        "planned",
        "notes":         "Run when weights found",
    },
    "lsda": {
        "factory":       _vis_dfb("lsda"),
        "detector_name": "LSDA",
        "modality":      "visual",
        "score_type":    "probability",
        "status":        "planned",
        "notes":         "Technical narrative only this week; CSV pending",
    },
}


# ---- rPPG (modality = "rppg") --------------------------------------------

def _rppg_pos(config: dict):
    from .rppg_detector import RPPGDetector
    return RPPGDetector(config)


def _rppg_tscan(config: dict):
    from .rppg_tscan_detector import TSCANDetector
    return TSCANDetector(config)


def _rppg_physmamba(config: dict):
    from .rppg_physmamba_detector import PhysMambaDetector
    return PhysMambaDetector(config)


RPPG_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── Tested ────────────────────────────────────────────────────────────
    "pos": {
        "factory":       _rppg_pos,
        "detector_name": "POS",
        "modality":      "rppg",
        "score_type":    "snr",
        "status":        "available",
        "notes":         "Wang 2017; no checkpoint; calibrated on FF++ val",
    },
    # ── Backup (confirm weights, then run sweep CSV) ──────────────────────
    "tscan": {
        "factory":       _rppg_tscan,
        "detector_name": "TS-CAN",
        "modality":      "rppg",
        "score_type":    "snr",
        "status":        "rppg_toolbox",
        "notes":         "rPPG-Toolbox TS-CAN; pending weight confirmation",
    },
    # ── Planned ───────────────────────────────────────────────────────────
    "physmamba": {
        "factory":       _rppg_physmamba,
        "detector_name": "PhysMamba",
        "modality":      "rppg",
        "score_type":    "snr",
        "status":        "planned",
        "notes":         "GP validator-approved alternative; pending implementation",
    },
}


# ---- AV Sync (modality = "sync") ----------------------------------------

def _sync_syncnet(config: dict):
    from .sync_detector import SyncDetector
    return SyncDetector(config)


def _sync_stub(name: str, notes: str = ""):
    def factory(config: dict):
        raise NotImplementedError(
            f"Sync detector '{name}' is planned but not yet implemented.\n"
            + (f"Notes: {notes}\n" if notes else "")
            + "See docs/v5_implementation_report.md §5.3 for roadmap."
        )
    return factory


SYNC_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── Tested (DFDC, DF-TIMIT confirmed) ────────────────────────────────
    "syncnet": {
        "factory":       _sync_syncnet,
        "detector_name": "SyncNet",
        "modality":      "av_sync",
        "score_type":    "sync_error",
        "status":        "available",
        "notes":         "LatentSync SyncNet-MDS; FakeAVCeleb pending dataset access",
    },
    # ── Backup / investigate ─────────────────────────────────────────────
    "mrdf": {
        "factory":       _sync_stub("mrdf", "needs training; skipped this week"),
        "detector_name": "MRDF",
        "modality":      "av_sync",
        "score_type":    "sync_error",
        "status":        "planned",
        "notes":         "Needs self-training; not prioritised for this sprint",
    },
    "latentsync": {
        "factory":       _sync_stub("latentsync", "investigation phase"),
        "detector_name": "LatentSync",
        "modality":      "av_sync",
        "score_type":    "sync_error",
        "status":        "planned",
        "notes":         "Alternative to SyncNet using LatentSync pipeline",
    },
    "avad": {
        "factory":       _sync_stub("avad", "investigation phase"),
        "detector_name": "AVAD",
        "modality":      "av_sync",
        "score_type":    "sync_error",
        "status":        "planned",
        "notes":         "Audio-Visual Anomaly Detection; under investigation",
    },
}


# ---- Combined registry and factory ---------------------------------------

REGISTRY: Dict[str, Dict[str, Dict[str, Any]]] = {
    "visual": VISUAL_REGISTRY,
    "rppg":   RPPG_REGISTRY,
    "sync":   SYNC_REGISTRY,
}

# Default detector per modality (matches GP-tested primary)
DEFAULTS: Dict[str, str] = {
    "visual": "xception",
    "rppg":   "pos",
    "sync":   "syncnet",
}


def build_detector(
    modality: str,
    detector_key: str,
    config: dict,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Instantiate a detector by modality and key.

    Returns (detector_instance, metadata_dict).
    metadata_dict contains: detector_name, modality, score_type, status, notes.

    Raises:
        KeyError          — unknown modality or detector_key
        NotImplementedError — detector is planned but not yet implemented
        ImportError       — required external repo (DFB, rPPG-Toolbox) not found
    """
    reg = REGISTRY.get(modality)
    if reg is None:
        raise KeyError(
            f"Unknown modality '{modality}'. Available: {sorted(REGISTRY)}"
        )
    entry = reg.get(detector_key)
    if entry is None:
        raise KeyError(
            f"Unknown {modality} detector '{detector_key}'. "
            f"Available: {sorted(reg)}"
        )
    detector = entry["factory"](config)
    meta = {k: v for k, v in entry.items() if k != "factory"}
    return detector, meta


def list_detectors(modality: str = None) -> Dict[str, Any]:
    """Return a summary of registered detectors, optionally filtered by modality."""
    if modality:
        return {k: {f: v[f] for f in ("detector_name", "status", "notes")}
                for k, v in REGISTRY.get(modality, {}).items()}
    return {
        mod: {k: {f: v[f] for f in ("detector_name", "status", "notes")}
              for k, v in reg.items()}
        for mod, reg in REGISTRY.items()
    }
