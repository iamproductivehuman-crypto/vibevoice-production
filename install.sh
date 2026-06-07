#!/bin/bash
# =============================================================================
#  install.sh — VibeVoice Production Installer
#  Target: fresh Vast.ai CUDA 12.x instance (RTX 3090 / RTX 4090 recommended)
#
#  Usage:
#      git clone https://github.com/iamproductivehuman-crypto/vibevoice-production.git
#      cd vibevoice-production
#      bash install.sh                     # install only
#      bash install.sh --start-server      # install and start API automatically
#
#  Environment:
#      AUTO_START_SERVER=1 bash install.sh  # same as --start-server
#      API_KEY=mysecret   bash install.sh   # set auth key before starting
#      PORT=8000          bash install.sh   # override server port (default: 8000)
# =============================================================================
set -euo pipefail

# ── Parse flags ───────────────────────────────────────────────────────────────
START_SERVER="${AUTO_START_SERVER:-0}"
for arg in "$@"; do
    case "${arg}" in
        --start-server) START_SERVER=1 ;;
        *) echo "Unknown argument: ${arg}"; exit 1 ;;
    esac
done

# ── Pinned versions ───────────────────────────────────────────────────────────
REPO_URL="https://github.com/harry2141985/VibeVoice.git"
VIBE_COMMIT="4b9af0f4dab7f250ad5bb79bcb443315dc3bba54"

TORCH_VERSION="2.5.1"
TORCH_CUDA_TAG="cu121"      # matches CUDA 12.1 driver (also works on 12.x)
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"

TRANSFORMERS_VERSION="4.51.3"
ACCELERATE_MIN="1.6.0"

MODEL_ID="microsoft/VibeVoice-1.5B"
VRAM_WARN_GB=16             # warn if GPU has less than this

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIBE_DIR="${SCRIPT_DIR}/VibeVoice"
LOG_FILE="${SCRIPT_DIR}/install.log"

# ── Logging helpers ───────────────────────────────────────────────────────────
log() { echo "$*" | tee -a "${LOG_FILE}"; }
die() { echo ""; echo "ERROR: $*" | tee -a "${LOG_FILE}"; exit 1; }

echo "" > "${LOG_FILE}"   # reset log

log "=================================================="
log "  VibeVoice Production Installer"
log "  torch==${TORCH_VERSION}+${TORCH_CUDA_TAG}"
log "  transformers==${TRANSFORMERS_VERSION}"
log "  accelerate>=${ACCELERATE_MIN}"
log "  repo: ${REPO_URL}"
log "  commit: ${VIBE_COMMIT}"
log "=================================================="

# ── Step 0 — GPU check ────────────────────────────────────────────────────────
log ""
log "[0/9] GPU check..."

command -v nvidia-smi &>/dev/null || die "nvidia-smi not found. This installer requires a CUDA GPU."

GPU_NAME=$(nvidia-smi --query-gpu=name         --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi  --query-gpu=memory.total --format=csv,noheader | head -1)
CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+" || echo "unknown")

log "  GPU:  ${GPU_NAME}"
log "  VRAM: ${GPU_MEM}"
log "  CUDA: ${CUDA_VER}"

VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
VRAM_GB=$(( VRAM_MIB / 1024 ))

if (( VRAM_GB < VRAM_WARN_GB )); then
    log ""
    log "  ⚠  WARNING: Your GPU has ~${VRAM_GB} GB VRAM."
    log "  ⚠  Recommended: RTX 3090 / RTX 4090 (24 GB)."
    log "  ⚠  Generation may fail with CUDA OOM on smaller cards."
    log "  ⚠  Continuing anyway — you can still try."
    log ""
fi

# ── Step 1 — Clone VibeVoice ─────────────────────────────────────────────────
log ""
log "[1/9] Cloning VibeVoice fork (harry2141985) @ ${VIBE_COMMIT}..."

if [ ! -d "${VIBE_DIR}" ]; then
    git clone "${REPO_URL}" "${VIBE_DIR}" 2>&1 | tee -a "${LOG_FILE}"
else
    log "  VibeVoice directory already exists — fetching updates."
    git -C "${VIBE_DIR}" fetch origin 2>&1 | tee -a "${LOG_FILE}"
fi

log "  Pinning to commit: ${VIBE_COMMIT}"
git -C "${VIBE_DIR}" checkout "${VIBE_COMMIT}" 2>&1 | tee -a "${LOG_FILE}"

ACTUAL_COMMIT="$(git -C "${VIBE_DIR}" rev-parse HEAD)"
log "  Actual commit: ${ACTUAL_COMMIT}"

# Verify critical file exists before wasting time on dependencies
INFERENCE_FILE="${VIBE_DIR}/vibevoice/modular/modeling_vibevoice_inference.py"
if [ ! -f "${INFERENCE_FILE}" ]; then
    log ""
    log "  Searching for inference implementation in this commit..."
    FOUND=$(find "${VIBE_DIR}/vibevoice" -name "*.py" \
            | xargs grep -l "ForConditionalGeneration" 2>/dev/null || true)
    if [ -n "${FOUND}" ]; then
        log "  Candidate files:"
        echo "${FOUND}" | while read -r f; do log "    ${f}"; done
        die "modeling_vibevoice_inference.py not found in commit ${ACTUAL_COMMIT}.
  This commit does not contain the required inference file.
  Fix: set VIBE_COMMIT to a SHA that includes that file, or use the harry2141985 fork.
  Candidates above may be the renamed equivalent — update imports if so."
    else
        die "modeling_vibevoice_inference.py not found and no ForConditionalGeneration class found.
  Wrong repo or wrong commit. Check REPO_URL and VIBE_COMMIT in install.sh."
    fi
fi
log "  ✓ modeling_vibevoice_inference.py present"

# ── Step 2 — PyTorch ─────────────────────────────────────────────────────────
log ""
log "[2/9] Installing PyTorch ${TORCH_VERSION}+${TORCH_CUDA_TAG}..."
pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision" \
    "torchaudio" \
    --index-url "${TORCH_INDEX}" \
    2>&1 | tee -a "${LOG_FILE}"

python - <<'PYEOF' 2>&1 | tee -a "${LOG_FILE}"
import torch, sys
assert torch.cuda.is_available(), "torch.cuda.is_available() is False after install!"
print(f"  torch {torch.__version__}  CUDA {torch.version.cuda}  device: {torch.cuda.get_device_name(0)}")
PYEOF

# ── Step 3 — All Python dependencies ─────────────────────────────────────────
log ""
log "[3/9] Installing Python dependencies (transformers, accelerate, FastAPI, uvicorn, ...)..."
pip install \
    "transformers==${TRANSFORMERS_VERSION}" \
    "accelerate>=${ACCELERATE_MIN}" \
    "soundfile>=0.12.1" \
    "librosa>=0.10.0" \
    "numpy<2.0" \
    "paramiko>=3.0" \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.29" \
    "python-multipart>=0.0.9" \
    "aiofiles>=23.0" \
    2>&1 | tee -a "${LOG_FILE}"

# Hard-check transformers version — a wrong version causes cryptic errors later
python - <<PYEOF 2>&1 | tee -a "${LOG_FILE}"
import transformers, sys
got  = transformers.__version__
want = "${TRANSFORMERS_VERSION}"
if got != want:
    print(f"  ERROR: transformers {got} installed but {want} required.")
    print(f"  Run: pip install transformers=={want}")
    sys.exit(1)
print(f"  transformers {got}  ✓")
PYEOF

# Quick check FastAPI + uvicorn are importable
python - <<'PYEOF' 2>&1 | tee -a "${LOG_FILE}"
import fastapi, uvicorn
print(f"  fastapi {fastapi.__version__}  uvicorn {uvicorn.__version__}  ✓")
PYEOF

# ── Step 4 — VibeVoice editable install ───────────────────────────────────────
log ""
log "[4/9] Installing VibeVoice package (pip install -e .)..."
cd "${VIBE_DIR}"
pip install -e . 2>&1 | tee -a "${LOG_FILE}"
cd "${SCRIPT_DIR}"

python - <<'PYEOF' 2>&1 | tee -a "${LOG_FILE}"
import vibevoice, sys
print(f"  vibevoice imported OK  (path: {vibevoice.__file__})")
PYEOF

# ── Step 5 — Patches ──────────────────────────────────────────────────────────
log ""
log "[5/9] Applying source patches (flash_attention_2 → sdpa, etc.)..."
python "${SCRIPT_DIR}/patch.py" 2>&1 | tee -a "${LOG_FILE}"

# ── Step 6 — Copy voices ──────────────────────────────────────────────────────
log ""
log "[6/9] Copying voices..."
mkdir -p "${VIBE_DIR}/demo/voices"
VOICES_SRC="${SCRIPT_DIR}/voices"
if ls "${VOICES_SRC}"/*.{wav,mp3,flac,ogg} 2>/dev/null | grep -q .; then
    cp "${VOICES_SRC}"/*.{wav,mp3,flac,ogg} "${VIBE_DIR}/demo/voices/" 2>/dev/null || true
    COUNT=$(ls "${VIBE_DIR}/demo/voices" | wc -l)
    log "  ${COUNT} voice file(s) copied."
else
    log "  No custom voices in ${VOICES_SRC}/ — add .wav/.mp3 files before generating."
    log "  You can also upload voices via the browser UI after the server starts."
fi

# ── Step 7 — Download model ───────────────────────────────────────────────────
log ""
log "[7/9] Downloading model ${MODEL_ID} (may take 5-15 min on first run)..."
python - <<PYEOF 2>&1 | tee -a "${LOG_FILE}"
import sys, torch
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference

print("  Loading model to verify download...")
try:
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        "${MODEL_ID}",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    print("  Model loaded OK — unloading...")
    del model
    torch.cuda.empty_cache()
    print("  Model unloaded.")
except Exception as e:
    print(f"  ERROR loading model: {e}")
    sys.exit(1)
PYEOF

# ── Step 8 — Test generation ──────────────────────────────────────────────────
log ""
log "[8/9] Running test generation (validate_install.py)..."
python "${SCRIPT_DIR}/validate_install.py" 2>&1 | tee -a "${LOG_FILE}"

# ── Step 9 — Full environment report ──────────────────────────────────────────
log ""
log "[9/9] Final environment report (verify.py)..."
python "${SCRIPT_DIR}/verify.py" 2>&1 | tee -a "${LOG_FILE}"

# ── Resolve public IP ─────────────────────────────────────────────────────────
SERVER_PORT="${PORT:-8000}"
SERVER_IP=$(curl -s --max-time 3 https://api.ipify.org 2>/dev/null \
    || curl -s --max-time 3 http://checkip.amazonaws.com 2>/dev/null \
    || hostname -I 2>/dev/null | awk '{print $1}' \
    || echo "localhost")

log ""
log "=================================================="
log "  ✅ Installation complete!"
log "  Log saved to: ${LOG_FILE}"
log ""
log "  Server running on:"
log "    http://${SERVER_IP}:${SERVER_PORT}"
log ""
log "  Endpoints:"
log "    GET  /health"
log "    GET  /voices"
log "    GET  /ui                   (browser UI)"
log "    POST /generate"
log "    POST /generate_url"
log "    POST /batch_generate"
log "    POST /upload_voice"
log ""
log "  Quick test:"
log "    curl http://${SERVER_IP}:${SERVER_PORT}/health"
log ""
log "  Batch generation (CLI):"
log "    python batch_generate.py --list_voices"
log "    python batch_generate.py \\"
log "      --input_dir  /path/to/txts \\"
log "      --output_dir /path/to/output \\"
log "      --speaker    Alice"
log ""
log "  Start server manually:"
log "    bash start_server.sh"
log "=================================================="

# ── Auto-start server ─────────────────────────────────────────────────────────
if [ "${START_SERVER}" = "1" ]; then
    log ""
    log "  --start-server flag detected — starting API server now..."
    log ""
    exec bash "${SCRIPT_DIR}/start_server.sh"
fi