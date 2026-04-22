"""
Anti-Deepfake-Box — Checkpoint Downloader
Downloads / prepares the three model weights needed to run the full pipeline.

Usage:
    python download_checkpoints.py

What this script does:
    1. physnet_ubfc.pth    — PhysNet (rPPG), copied from third_party/rPPG-Toolbox
                             or cloned on-the-fly from GitHub
    2. latentsync_syncnet.pth   — LatentSync SyncNet, downloaded from HuggingFace
                             ByteDance/LatentSync-1.6
    3. xception_ff_c23.pth — XceptionNet, ImageNet pretrained base (timm).
                             NOTE: this is NOT the FF++ fine-tuned version.
                             For full accuracy, replace with the FF++ checkpoint
                             obtained via the official FaceForensics++ request form.
                             The system works with this base version — it uses the
                             same architecture and backbone features, just not
                             fine-tuned on deepfake data.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
CHECKPOINTS = ROOT / "checkpoints"
THIRD_PARTY = ROOT / "third_party"
CHECKPOINTS.mkdir(exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def ok(msg):  print(f"  ✓ {msg}")
def info(msg): print(f"  → {msg}")
def warn(msg): print(f"  ⚠ {msg}")
def header(msg): print(f"\n[{msg}]")


def pip_install(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


# ─────────────────────────────────────────────────────────────────────────────
# 1. PhysNet (MA-UBFC pretrained)
# ─────────────────────────────────────────────────────────────────────────────

def download_physnet():
    header("1/3  PhysNet  →  checkpoints/physnet_ubfc.pth")
    dst = CHECKPOINTS / "physnet_ubfc.pth"
    if dst.exists():
        ok(f"already exists ({dst.stat().st_size / 1e6:.1f} MB)")
        return

    # Option A: already cloned via setup.sh
    src = THIRD_PARTY / "rPPG-Toolbox" / "final_model_release" / "MA-UBFC_physnet.pth"
    if src.exists():
        shutil.copy(src, dst)
        ok(f"copied from {src} ({dst.stat().st_size / 1e6:.1f} MB)")
        return

    # Option B: clone rPPG-Toolbox now (depth 1 for speed)
    info("Cloning rPPG-Toolbox (depth 1)...")
    THIRD_PARTY.mkdir(exist_ok=True)
    rppg_dir = THIRD_PARTY / "rPPG-Toolbox"
    if not rppg_dir.exists():
        subprocess.check_call([
            "git", "clone", "--depth", "1",
            "https://github.com/nia1003/rppg-toolbox.git",
            str(rppg_dir),
        ])
    shutil.copy(src, dst)
    ok(f"downloaded and copied ({dst.stat().st_size / 1e6:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SyncNet (ByteDance/LatentSync-1.6)
# ─────────────────────────────────────────────────────────────────────────────

def download_syncnet():
    header("2/3  SyncNet  →  checkpoints/latentsync_syncnet.pth")
    dst = CHECKPOINTS / "latentsync_syncnet.pth"
    if dst.exists():
        ok(f"already exists ({dst.stat().st_size / 1e6:.1f} MB)")
        return

    info("Installing huggingface_hub...")
    pip_install("huggingface_hub")

    from huggingface_hub import hf_hub_download
    info("Downloading from ByteDance/LatentSync-1.6 on HuggingFace...")
    path = hf_hub_download(
        repo_id="ByteDance/LatentSync-1.6",
        filename="latentsync_syncnet.pth",
        local_dir=str(CHECKPOINTS),
    )
    ok(f"downloaded to {path} ({Path(path).stat().st_size / 1e6:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# 3. XceptionNet (ImageNet base → 2-class head)
# ─────────────────────────────────────────────────────────────────────────────

def download_xception():
    header("3/3  XceptionNet  →  checkpoints/xception_ff_c23.pth")
    dst = CHECKPOINTS / "xception_ff_c23.pth"
    if dst.exists():
        ok(f"already exists ({dst.stat().st_size / 1e6:.1f} MB)")
        return

    info("Installing timm...")
    pip_install("timm")

    import torch
    import torch.nn as nn
    import timm

    info("Loading xception (ImageNet pretrained) from timm...")
    model = timm.create_model("xception", pretrained=True)

    # Replace final classifier with 2-class head (real=0, fake=1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    nn.init.xavier_normal_(model.fc.weight)
    nn.init.zeros_(model.fc.bias)

    # Save in the format visual_detector.py expects
    torch.save(model.state_dict(), dst)
    ok(f"saved ({dst.stat().st_size / 1e6:.1f} MB)")
    warn("This is an ImageNet base — backbone features are good, but the")
    warn("2-class head is randomly initialized (not FF++ fine-tuned).")
    warn("For full deepfake detection accuracy, replace with the FF++ weight")
    warn("from: https://github.com/ondyari/FaceForensics (requires registration)")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary():
    print("\n" + "=" * 55)
    print("Checkpoints ready:")
    files = {
        "physnet_ubfc.pth":    "PhysNet rPPG detector (UBFC pretrained)",
        "latentsync_syncnet.pth":   "LatentSync SyncNet (ByteDance, 94% acc)",
        "xception_ff_c23.pth": "XceptionNet 2-class (ImageNet base *)",
    }
    for fname, desc in files.items():
        p = CHECKPOINTS / fname
        status = f"{p.stat().st_size / 1e6:.1f} MB" if p.exists() else "MISSING"
        print(f"  {fname:<30} {status:<10}  {desc}")

    print("\n* Replace xception_ff_c23.pth with the FF++ fine-tuned version")
    print("  for best visual detection accuracy.")
    print("\nNext step:")
    print("  export PYTHONPATH=$(pwd):third_party/rPPG-Toolbox:third_party/FaceForensics/classification:third_party/LatentSync")
    print("  python scripts/inference.py --video <your_video.mp4> --config configs/default.yaml")
    print("=" * 55)


if __name__ == "__main__":
    print("Anti-Deepfake-Box — downloading checkpoints...")
    try:
        download_physnet()
    except Exception as e:
        warn(f"PhysNet failed: {e}")

    try:
        download_syncnet()
    except Exception as e:
        warn(f"SyncNet failed: {e}")

    try:
        download_xception()
    except Exception as e:
        warn(f"XceptionNet failed: {e}")

    print_summary()
