"""
Auto-detect hardware capabilities and select the appropriate inference profile.

Profiles (ordered by capability):
  cloud       — full 4-path pipeline, meta-classifier fusion
  jetson_orin — GPU: all 3 detectors + FFT; PhysNet + SyncNet enabled
  jetson_nano — limited GPU: Visual + CHROM rPPG + FFT (no SyncNet)
  cpu_only    — CPU: CHROM rPPG + FFT only, no heavy neural models
  browser     — minimal API server (1 fps, FFT + lightweight face check)
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class HardwareInfo:
    cpu_cores: int
    ram_gb: float
    has_cuda: bool
    gpu_name: Optional[str]
    vram_gb: Optional[float]
    is_jetson: bool
    jetson_model: Optional[str]   # e.g. "NVIDIA Jetson Orin Nano"

    @property
    def profile_name(self) -> str:
        if self.has_cuda and self.vram_gb and self.vram_gb >= 6.0:
            return "cloud"
        if self.is_jetson and self.vram_gb and self.vram_gb >= 4.0:
            return "jetson_orin"
        if self.is_jetson or (self.has_cuda and self.vram_gb and self.vram_gb >= 1.0):
            return "jetson_nano"
        return "cpu_only"


def detect_hardware() -> HardwareInfo:
    import multiprocessing

    cpu_cores = multiprocessing.cpu_count()

    # RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1e9
    except ImportError:
        ram_gb = _parse_meminfo()

    # CUDA / GPU
    has_cuda, gpu_name, vram_gb = False, None, None
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    except ImportError:
        pass

    # Jetson detection
    is_jetson = False
    jetson_model = None
    jetson_path = Path("/proc/device-tree/model")
    if jetson_path.exists():
        try:
            model_str = jetson_path.read_text()
            if "jetson" in model_str.lower() or "nvidia" in model_str.lower():
                is_jetson = True
                jetson_model = model_str.strip("\x00").strip()
        except Exception:
            pass

    return HardwareInfo(
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        has_cuda=has_cuda,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        is_jetson=is_jetson,
        jetson_model=jetson_model,
    )


def _parse_meminfo() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return 4.0  # conservative fallback


# ── Profile loading ──────────────────────────────────────────────────────────

PROFILES_DIR = Path(__file__).parent.parent / "configs" / "profiles"


def load_profile(name: Optional[str] = None) -> dict:
    """
    Load a hardware profile YAML.
    If name is None, auto-detect from hardware.
    Returns merged config (profile overrides default.yaml).
    """
    if name is None:
        hw = detect_hardware()
        name = hw.profile_name
        _print_hw_summary(hw)

    profile_path = PROFILES_DIR / f"{name}.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    # Load default first, then overlay profile
    default_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    cfg = yaml.safe_load(default_path.read_text()) if default_path.exists() else {}
    profile_cfg = yaml.safe_load(profile_path.read_text())
    _deep_merge(cfg, profile_cfg)

    print(f"[ADB-Edge] Profile: {name}")
    return cfg


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _print_hw_summary(hw: HardwareInfo) -> None:
    print(f"[ADB-Edge] Hardware detected:")
    print(f"  CPU cores : {hw.cpu_cores}")
    print(f"  RAM       : {hw.ram_gb:.1f} GB")
    if hw.has_cuda:
        print(f"  GPU       : {hw.gpu_name} ({hw.vram_gb:.1f} GB VRAM)")
    else:
        print(f"  GPU       : None (CPU-only mode)")
    if hw.is_jetson:
        print(f"  Platform  : {hw.jetson_model}")
    print(f"  → Profile : {hw.profile_name}")
