#!/bin/bash
set -euo pipefail

# ── Configuration ─────────────────────────────────────
REPO_URL="https://github.com/harry2141985/VibeVoice.git"
# Pin to a specific commit for reproducibility.
# Update this hash when you verify a newer commit works.
VIBE_COMMIT="HEAD"   # replace with exact SHA once confirmed, e.g. "a3f8c21"

TRANSFORMERS_VERSION="4.51.3"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================================="
echo "  VibeVoice Production Setup"
echo "  Repo:   ${REPO_URL}"
echo "  Commit: ${VIBE_COMMIT}"
echo "=================================================="

# ── 0. GPU sanity check ───────────────────────────────
echo ""
echo "[0/7] Checking GPU..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. This script requires a CUDA-capable GPU."
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "  GPU OK"

# ── 1. Clone VibeVoice ────────────────────────────────
echo ""
echo "[1/7] Cloning VibeVoice (fork: harry2141985)..."
if [ ! -d "VibeVoice" ]; then
    git clone "${REPO_URL}" VibeVoice
else
    echo "  VibeVoice directory exists — pulling latest."
    git -C VibeVoice fetch origin
fi

cd VibeVoice

if [ "${VIBE_COMMIT}" != "HEAD" ]; then
    echo "  Checking out pinned commit: ${VIBE_COMMIT}"
    git checkout "${VIBE_COMMIT}"
else
    echo "  WARNING: Using HEAD. Pin VIBE_COMMIT to a SHA for reproducible builds."
    git log -1 --oneline
fi

ACTUAL_COMMIT="$(git rev-parse HEAD)"
echo "  Commit: ${ACTUAL_COMMIT}"
cd ..

# ── 2. Install PyTorch (CUDA 12.1) ───────────────────
echo ""
echo "[2/7] Installing PyTorch (cu121)..."
pip install -q \
    torch \
    torchvision \
    torchaudio \
    --index-url "${TORCH_INDEX}"

# ── 3. Pin transformers + core deps ──────────────────
echo ""
echo "[3/7] Installing pinned dependencies..."
pip install -q \
    "transformers==${TRANSFORMERS_VERSION}" \
    "accelerate>=0.27.0" \
    "soundfile>=0.12.1" \
    "librosa>=0.10.0" \
    "numpy<2.0" \
    "paramiko>=3.0"

# ── 4. Install VibeVoice in editable mode ─────────────
echo ""
echo "[4/7] Installing VibeVoice package (pip install -e .)..."
cd VibeVoice
pip install -q -e .
cd ..

# ── 5. Apply patches ──────────────────────────────────
echo ""
echo "[5/7] Applying patches..."
cd VibeVoice
python "${SCRIPT_DIR}/patch.py"
cd ..

# ── 6. Copy voices ────────────────────────────────────
echo ""
echo "[6/7] Copying voices..."
mkdir -p VibeVoice/demo/voices
if ls "${SCRIPT_DIR}/voices/"*.{wav,mp3,flac,ogg} 2>/dev/null | grep -q .; then
    cp "${SCRIPT_DIR}"/voices/*.{wav,mp3,flac,ogg} VibeVoice/demo/voices/ 2>/dev/null || true
    echo "  Voices copied."
else
    echo "  No custom voices found in ${SCRIPT_DIR}/voices/ — using defaults."
fi

# ── 7. Verify ─────────────────────────────────────────
echo ""
echo "[7/7] Running verification..."
python "${SCRIPT_DIR}/verify.py"

echo ""
echo "=================================================="
echo "  Setup complete!"
echo "  Run: python batch_generate.py --list_voices"
echo "  Run: python batch_generate.py \\"
echo "         --input_dir  /path/to/txts \\"
echo "         --output_dir /path/to/output \\"
echo "         --speaker    Alice"
echo "=================================================="