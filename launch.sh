#!/bin/bash
# =============================================================================
#  launch.sh — One-command VibeVoice launcher
#
#  This is the single entry point for running VibeVoice on a fresh instance.
#  It verifies the installation, then starts the server on port 8000.
#
#  Usage:
#      bash launch.sh              # foreground (Ctrl-C to stop)
#      bash launch.sh --daemon     # background (nohup, logs to server.log)
#
#  If VibeVoice is not installed, runs install.sh first automatically.
#
#  Environment variables:
#      PORT=8000                   # override listen port (default: 8000)
#      CLEANUP_RETENTION_HOURS=24  # hours to keep generated WAV files
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

PORT="${PORT:-8000}"

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║         VibeVoice Studio — Launch               ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# ── Step 1: Check installation ───────────────────────────────────────────────
if [ ! -d "${SCRIPT_DIR}/VibeVoice" ]; then
    echo -e "${YELLOW}⚠  VibeVoice is not installed. Running install.sh first...${NC}"
    echo ""
    bash "${SCRIPT_DIR}/install.sh"
    echo ""
fi

# ── Step 2: Check Python environment ─────────────────────────────────────────
if ! python -c "import fastapi, uvicorn, torch" 2>/dev/null; then
    echo -e "${YELLOW}⚠  Some Python dependencies missing. Running install.sh...${NC}"
    bash "${SCRIPT_DIR}/install.sh"
fi

# ── Step 3: Validate VibeVoice import ────────────────────────────────────────
export PYTHONPATH="${SCRIPT_DIR}/VibeVoice:${PYTHONPATH:-}"
if ! python -c "from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference" 2>/dev/null; then
    echo -e "${RED}✗  VibeVoice import failed. Try running: bash install.sh${NC}"
    exit 1
fi
echo -e "${GREEN}✓  VibeVoice installed and importable${NC}"

# ── Step 4: Check GPU ─────────────────────────────────────────────────────────
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "unknown")
    echo -e "${GREEN}✓  GPU detected: ${GPU_NAME}${NC}"
else
    echo -e "${YELLOW}⚠  No CUDA GPU detected. Generation will be very slow on CPU.${NC}"
fi

# ── Step 5: Check model cache ────────────────────────────────────────────────
if python -c "
from pathlib import Path
import os
model = os.getenv('VIBEVOICE_MODEL', 'microsoft/VibeVoice-1.5B')
cache = Path.home() / '.cache' / 'huggingface' / 'hub'
slug = 'models--' + model.replace('/', '--')
cached = (cache / slug).is_dir()
exit(0 if cached else 1)
" 2>/dev/null; then
    echo -e "${GREEN}✓  Model found in HuggingFace cache${NC}"
else
    echo -e "${YELLOW}⚠  Model not cached — will download on first request (~6 GB)${NC}"
fi

# ── Step 6: Check static UI ──────────────────────────────────────────────────
if [ -f "${SCRIPT_DIR}/static/index.html" ]; then
    echo -e "${GREEN}✓  Static UI found${NC}"
else
    echo -e "${YELLOW}⚠  static/index.html not found — UI may not load${NC}"
fi

echo ""
echo -e "${GREEN}All checks passed. Starting server on port ${PORT}...${NC}"
echo ""

# ── Step 7: Print access instructions ────────────────────────────────────────
# Detect SSH port for tunnel instructions
SSH_PORT=$(grep -E "^Port " /etc/ssh/sshd_config 2>/dev/null | awk '{print $2}' || echo "<PORT>")
if [ -z "${SSH_PORT}" ]; then SSH_PORT="<PORT>"; fi

echo "================================================"
echo " VibeVoice Ready"
echo "================================================"
echo ""
echo " Local UI:    http://localhost:${PORT}/ui/"
echo " Health:      http://localhost:${PORT}/health"
echo " Docs:        http://localhost:${PORT}/docs"
echo ""
echo " If using SSH tunnel, run this on your LOCAL machine:"
echo ""
echo "   ssh -p ${SSH_PORT} root@<HOST> -L ${PORT}:localhost:${PORT}"
echo ""
echo " Then open:   http://localhost:${PORT}/ui/"
echo ""
echo "================================================"
echo ""

# ── Step 8: Delegate to start_server.sh ──────────────────────────────────────
exec bash "${SCRIPT_DIR}/start_server.sh" "$@"
