"""Config loading utilities with mode overlay support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).parent


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base; overlay values win."""
    result = dict(base)
    for k, v in overlay.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: str = "configs/default.yaml",
                mode: str | None = None) -> dict:
    """
    Load config YAML, optionally overlaying a mode-specific config.

    Parameters
    ----------
    config_path : base config file (default: configs/default.yaml)
    mode        : "forensic" | "realtime" | None
                  Loads configs/mode_{mode}.yaml and merges on top of base.

    Usage
    -----
    cfg = load_config("configs/default.yaml", mode="forensic")
    cfg = load_config("configs/default.yaml", mode="realtime")
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    if mode:
        mode_path = _CONFIG_DIR / f"mode_{mode}.yaml"
        if not mode_path.exists():
            raise FileNotFoundError(
                f"Mode config not found: {mode_path}\n"
                f"Available modes: forensic, realtime"
            )
        with open(mode_path) as f:
            overlay = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, overlay)
        cfg["_mode"] = mode

    return cfg
