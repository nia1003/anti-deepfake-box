"""
Anti-Deepfake-Box — Checkpoint Downloader
Downloads / prepares the two model weights needed to run the full pipeline.

Usage:
    python download_checkpoints.py

What this script does:
    1. latentsync_syncnet.pth — LatentSync SyncNet, downloaded from HuggingFace
                                ByteDance/LatentSync-1.6 (file: stable_syncnet.pt)
    2. xception_ff_c23.pth   — XceptionNet, ImageNet pretrained base (timm).
                                NOTE: this is NOT the FF++ fine-tuned version.
                                For full accuracy, replace with the FF++ checkpoint
                                obtained via the official FaceForensics++ request form.

Note: rPPG detector uses POS (Plane-Orthogonal-to-Skin), a pure NumPy/SciPy
      algorithm — no checkpoint required.
"""

import sys
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
CHECKPOINTS = ROOT / "checkpoints"
CHECKPOINTS.mkdir(exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def ok(msg):     print(f"  ✓ {msg}")
def info(msg):   print(f"  → {msg}")
def warn(msg):   print(f"  ⚠ {msg}")
def header(msg): print(f"\n[{msg}]")


def pip_install(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


# ─────────────────────────────────────────────────────────────────────────────
# 1. SyncNet (ByteDance/LatentSync-1.6)
# ─────────────────────────────────────────────────────────────────────────────

def download_syncnet():
    header("1/2  SyncNet  →  checkpoints/latentsync_syncnet.pth")
    dst = CHECKPOINTS / "latentsync_syncnet.pth"
    if dst.exists():
        ok(f"already exists ({dst.stat().st_size / 1e6:.1f} MB)")
        return

    info("Installing huggingface_hub...")
    pip_install("huggingface_hub")

    from huggingface_hub import hf_hub_download
    # HuggingFace filename is "stable_syncnet.pt"; saved as "latentsync_syncnet.pth"
    info("Downloading stable_syncnet.pt from ByteDance/LatentSync-1.6...")
    path = hf_hub_download(
        repo_id="ByteDance/LatentSync-1.6",
        filename="stable_syncnet.pt",
        local_dir=str(CHECKPOINTS),
    )
    src = Path(path)
    if src != dst:
        shutil.copy(src, dst)
    ok(f"saved as latentsync_syncnet.pth ({dst.stat().st_size / 1e6:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. XceptionNet (ImageNet base → 2-class head)
# ─────────────────────────────────────────────────────────────────────────────

def download_xception():
    header("2/2  XceptionNet  →  checkpoints/xception_ff_c23.pth")
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
    print("\n" + "=" * 60)
    print("Checkpoints ready:")
    files = {
        "latentsync_syncnet.pth": "LatentSync SyncNet (ByteDance, 94% acc)",
        "xception_ff_c23.pth":    "XceptionNet 2-class (ImageNet base *)",
    }
    for fname, desc in files.items():
        p = CHECKPOINTS / fname
        status = f"{p.stat().st_size / 1e6:.1f} MB" if p.exists() else "MISSING"
        print(f"  {fname:<28} {status:<10}  {desc}")

    print()
    print("  rPPG detector: POS (Wang 2017) — no checkpoint needed.")
    print()
    print("* Replace xception_ff_c23.pth with the FF++ fine-tuned version")
    print("  for best visual detection accuracy.")
    print("\nNext step:")
    print("  export PYTHONPATH=$(pwd):third_party/FaceForensics/classification:third_party/LatentSync")
    print("  python scripts/inference.py --video <your_video.mp4> --config configs/default.yaml")
    print("=" * 60)


if __name__ == "__main__":
    print("Anti-Deepfake-Box — downloading checkpoints...")

    try:
        download_syncnet()
    except Exception as e:
        warn(f"SyncNet failed: {e}")

    try:
        download_xception()
    except Exception as e:
        warn(f"XceptionNet failed: {e}")

    print_summary()
