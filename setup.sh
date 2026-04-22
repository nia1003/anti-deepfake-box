#!/usr/bin/env bash
# Anti-Deepfake-Box — one-shot setup script
# Usage: bash setup.sh
set -e

echo "=== Anti-Deepfake-Box Setup ==="

# ── 1. Python dependencies ───────────────────────────────────────────────────
echo "[1/4] Installing Python dependencies..."
pip install -r requirements.txt

# ── 2. Third-party repos ─────────────────────────────────────────────────────
echo "[2/4] Cloning third-party repos..."
mkdir -p third_party
cd third_party

if [ ! -d "rPPG-Toolbox" ]; then
    git clone --depth 1 https://github.com/nia1003/rppg-toolbox.git rPPG-Toolbox
else
    echo "  rPPG-Toolbox already cloned"
fi

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
echo "[3/4] Setting up checkpoints..."
mkdir -p checkpoints

# PhysNet — already in rPPG-Toolbox repo
PHYSNET_SRC="third_party/rPPG-Toolbox/final_model_release/MA-UBFC_physnet.pth"
if [ -f "$PHYSNET_SRC" ] && [ ! -f "checkpoints/physnet_ubfc.pth" ]; then
    ln -sf "$(pwd)/$PHYSNET_SRC" checkpoints/physnet_ubfc.pth
    echo "  physnet_ubfc.pth linked"
fi

# SyncNet — download from HuggingFace
if [ ! -f "checkpoints/stable_syncnet.pt" ]; then
    echo "  Downloading stable_syncnet.pt from HuggingFace..."
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('ByteDance/LatentSync-1.6', 'stable_syncnet.pt', local_dir='checkpoints')
print('  stable_syncnet.pt downloaded')
"
fi

# XceptionNet — must be provided manually
if [ ! -f "checkpoints/xception_ff_c23.pth" ]; then
    echo "  [optional] xception_ff_c23.pth not found — will use EfficientNet fallback"
    echo "  To enable: copy your FF++ XceptionNet checkpoint to checkpoints/xception_ff_c23.pth"
fi

# ── 4. PYTHONPATH hint ───────────────────────────────────────────────────────
echo "[4/4] Setup complete."
echo ""
echo "Set PYTHONPATH before running scripts:"
echo ""
echo "  export PYTHONPATH=\$(pwd):third_party/rPPG-Toolbox:third_party/FaceForensics/classification:third_party/LatentSync"
echo ""
echo "Quick start:"
echo "  python scripts/inference.py --video <your_video.mp4> --config configs/default.yaml"
echo "  ADB_PROFILE=cpu_only uvicorn api.app:app --host 0.0.0.0 --port 8000"
