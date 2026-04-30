"""
Anti-Deepfake-Box — Checkpoint Downloader
Downloads / prepares the model weights needed to run the full pipeline.

Usage:
    python download_checkpoints.py           # SyncNet + XceptionNet
    python download_checkpoints.py --tscan   # also show TS-CAN instructions

Checkpoints overview
--------------------
Name                        Required?  Source
latentsync_syncnet.pth      YES        HuggingFace: ByteDance/LatentSync-1.6
xception_ff_c23.pth         YES *      timm (ImageNet base) or FF++ form
tscan_ubfc.pth              optional   rppg-toolbox (see --tscan)
physnet_ubfc.pth            NOT USED   RPPGDetector uses POS algorithm (no file needed)

* The ImageNet base XceptionNet is downloaded automatically.
  For full FF++-fine-tuned accuracy, replace it via the FaceForensics++ form.
"""

import sys
import shutil
import subprocess
import argparse
from pathlib import Path

ROOT = Path(__file__).parent
CHECKPOINTS = ROOT / "checkpoints"
CHECKPOINTS.mkdir(exist_ok=True)


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
# 3. TS-CAN (rppg-toolbox, optional)
# ─────────────────────────────────────────────────────────────────────────────

def print_tscan_instructions():
    header("(optional)  TS-CAN  →  checkpoints/tscan_ubfc.pth")
    dst = CHECKPOINTS / "tscan_ubfc.pth"
    if dst.exists():
        ok(f"already exists ({dst.stat().st_size / 1e6:.1f} MB)")
        return

    print("""
  TS-CAN is a neural rPPG model trained on UBFC-rPPG.
  Without it, exp/detectors/tscan_detector.py falls back to the POS
  algorithm (no checkpoint needed) — detection still works.

  To use the full neural TS-CAN:

  Step 1: Clone rppg-toolbox
    git clone https://github.com/ubicomplab/rPPG-Toolbox.git

  Step 2: Download UBFC-rPPG pretrained weights
    The model zoo is linked in the rppg-toolbox README:
      https://github.com/ubicomplab/rPPG-Toolbox#model-zoo
    Download the TSCAN_UBFC-rPPG file and rename / place it as:
      checkpoints/tscan_ubfc.pth

  Step 3: Verify
    python -c "
    import torch
    s = torch.load('checkpoints/tscan_ubfc.pth', map_location='cpu')
    print('keys:', list(s.keys())[:5])
    "

  Note: the rppg-toolbox checkpoint format uses keys like
  'motion_conv1.weight', 'app_conv1.weight', etc. — matching the
  TSCAN architecture in exp/detectors/tscan_detector.py.
""")


# ─────────────────────────────────────────────────────────────────────────────
# Note: physnet_ubfc.pth is NOT needed
# ─────────────────────────────────────────────────────────────────────────────

def print_physnet_note():
    header("Note: physnet_ubfc.pth")
    print("""
  physnet_ubfc.pth was referenced in older config files but is NOT used.
  The RPPGDetector runs the POS algorithm (Wang 2017) — a pure NumPy/SciPy
  signal processing method that requires no model weights or GPU.

  You do NOT need to download physnet_ubfc.pth.
  The 'pretrained' key in rppg detector configs is intentionally ignored.
""")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary():
    print("\n" + "=" * 62)
    print("Checkpoints status:")
    files = {
        "latentsync_syncnet.pth": "LatentSync SyncNet (ByteDance)",
        "xception_ff_c23.pth":    "XceptionNet 2-class (ImageNet base *)",
        "tscan_ubfc.pth":         "TS-CAN UBFC (optional, POS fallback if absent)",
    }
    for fname, desc in files.items():
        p = CHECKPOINTS / fname
        if p.exists():
            status = f"{p.stat().st_size / 1e6:.1f} MB"
        else:
            status = "absent (optional)" if "tscan" in fname else "MISSING"
        print(f"  {fname:<28} {status:<22} {desc}")

    print()
    print("  rPPG / POS: no checkpoint needed (built-in NumPy algorithm).")
    print("  physnet_ubfc.pth: NOT required (stale config reference, ignored).")
    print()
    print("* Replace xception_ff_c23.pth with the FF++ fine-tuned version")
    print("  for best visual detection accuracy.")
    print("\nNext step:")
    print("  python exp/run_exp.py --detector all --dataset ff --ff_root /data/FF++")
    print("=" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Download Anti-Deepfake-Box checkpoints")
    p.add_argument(
        "--tscan", action="store_true",
        help="Show TS-CAN (rppg-toolbox) download instructions",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Anti-Deepfake-Box — downloading checkpoints...")

    try:
        download_syncnet()
    except Exception as e:
        warn(f"SyncNet failed: {e}")

    try:
        download_xception()
    except Exception as e:
        warn(f"XceptionNet failed: {e}")

    if args.tscan:
        print_tscan_instructions()

    print_physnet_note()
    print_summary()
