#!/usr/bin/env bash
# Anti-Deepfake-Box — one-shot setup script
# Usage: bash setup.sh
set -e

echo "=== Anti-Deepfake-Box Setup ==="

# ── 1. Python dependencies ───────────────────────────────────────────────────
echo "[1/3] Installing Python dependencies..."
pip install -r requirements.txt

# ── 2. Third-party repos ─────────────────────────────────────────────────────
echo "[2/3] Cloning third-party repos..."
mkdir -p third_party
cd third_party

if [ ! -d "FaceForensics" ]; then
    git clone --depth 1 https://github.com/nia1003/faceforensics.git FaceForensics
else
    echo "  FaceForensics already cloned"
fi

if [ ! -d "LatentSync" ]; then
    git clone --depth 1 https://github.com/nia1003/latentsync.git LatentSync
else
    echo "  LatentSync already cloned"
fi

cd ..

# ── 3. Checkpoints directory ─────────────────────────────────────────────────
echo "[3/3] Setting up checkpoints..."
mkdir -p checkpoints

# rPPG detector uses POS (Wang 2017) — pure NumPy/SciPy, no checkpoint needed.

# SyncNet — download from HuggingFace (hosted as stable_syncnet.pt)
if [ ! -f "checkpoints/latentsync_syncnet.pth" ]; then
    echo "  Downloading stable_syncnet.pt from HuggingFace..."
    python3 -c "
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
path = hf_hub_download('ByteDance/LatentSync-1.6', 'stable_syncnet.pt', local_dir='checkpoints')
dst = Path('checkpoints/latentsync_syncnet.pth')
if Path(path) != dst:
    shutil.copy(path, dst)
print('  latentsync_syncnet.pth ready')
"
fi

# XceptionNet — run download_checkpoints.py to auto-generate ImageNet base,
#               or replace with the FF++ fine-tuned weight for full accuracy.
if [ ! -f "checkpoints/xception_ff_c23.pth" ]; then
    echo "  [optional] xception_ff_c23.pth not found — run: python download_checkpoints.py"
fi

echo ""
echo "Setup complete. Set PYTHONPATH before running scripts:"
echo ""
echo "  export PYTHONPATH=\$(pwd):third_party/FaceForensics/classification:third_party/LatentSync"
echo ""
echo "Quick start:"
echo "  python scripts/inference.py --video <your_video.mp4> --config configs/default.yaml"
echo "  ADB_PROFILE=cpu_only uvicorn api.app:app --host 0.0.0.0 --port 8000"
