#!/bin/bash
# =============================================================================
#  start_server.sh — Start the VibeVoice API server
#
#  Loads the model once, then serves HTTP requests on port 8000.
#  Keep this running in a tmux session so it survives SSH disconnects.
#
#  Usage:
#      bash start_server.sh               # foreground (Ctrl-C to stop)
#      bash start_server.sh --daemon      # run in background via nohup
#
#  Environment variables:
#      PORT=8000                          # override listen port (default: 8000)
#      VIBEVOICE_HOST=0.0.0.0            # override listen host
#      CLEANUP_RETENTION_HOURS=24        # hours to keep generated WAV files
#
#  Remote access (SSH tunnel from your local machine):
#      ssh -p <VAST_SSH_PORT> root@<HOST> -L 8000:localhost:8000
#      Then open: http://localhost:8000/ui/
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

HOST="${VIBEVOICE_HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS=1          # always 1 — single GPU, model must not be forked
LOG_LEVEL="info"
DAEMON=false

# Parse arguments
for arg in "$@"; do
    case "${arg}" in
        --daemon) DAEMON=true ;;
        *) echo "Unknown argument: ${arg}"; exit 1 ;;
    esac
done

# Optional: activate the virtual environment if present
if [ -f "/venv/main/bin/activate" ]; then
    source /venv/main/bin/activate
fi

# Ensure VibeVoice package is on the path
VIBE_DIR="${SCRIPT_DIR}/VibeVoice"
if [ ! -d "${VIBE_DIR}" ]; then
    echo "ERROR: VibeVoice directory not found at ${VIBE_DIR}"
    echo "       Run 'bash install.sh' first."
    exit 1
fi
export PYTHONPATH="${VIBE_DIR}:${PYTHONPATH:-}"

# Check that FastAPI + uvicorn are installed
if ! python -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "Installing server dependencies (fastapi, uvicorn)..."
    pip install "fastapi>=0.115" "uvicorn[standard]>=0.29" --quiet
fi

# Detect SSH port from sshd_config (Vast.ai uses a non-standard port)
SSH_PORT=$(grep -E "^Port " /etc/ssh/sshd_config 2>/dev/null | awk '{print $2}' || echo "22")
if [ -z "${SSH_PORT}" ]; then
    SSH_PORT="22"
fi

# ── Startup banner ────────────────────────────────────────────────────────────
echo ""
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

UVICORN_CMD=(
    uvicorn server:app
    --host "${HOST}"
    --port "${PORT}"
    --workers "${WORKERS}"
    --log-level "${LOG_LEVEL}"
    --no-access-log
)

if [ "${DAEMON}" = "true" ]; then
    LOG_OUT="${SCRIPT_DIR}/server.log"
    echo -e "${CYAN}Starting in background. Logs: ${LOG_OUT}${NC}"
    nohup "${UVICORN_CMD[@]}" >> "${LOG_OUT}" 2>&1 &
    SERVER_PID=$!
    echo -e "${GREEN}Server PID: ${SERVER_PID}${NC}"
    echo -e "${GREEN}Stop with: kill ${SERVER_PID}${NC}"
    echo ""

    # Wait up to 15 seconds for the server to bind and respond
    echo -n "Waiting for server to start"
    for i in $(seq 1 30); do
        sleep 0.5
        if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
            echo ""
            echo -e "${GREEN}✓ Server is up and responding on port ${PORT}${NC}"
            echo ""
            echo -e "${CYAN}════════════════════════════════════════════════${NC}"
            echo -e "${CYAN}  Open: http://localhost:${PORT}/ui/${NC}"
            echo -e "${CYAN}════════════════════════════════════════════════${NC}"
            echo ""
            exit 0
        fi
        echo -n "."
    done
    echo ""
    echo -e "${YELLOW}⚠  Server did not respond within 15s. Check logs: tail -f ${LOG_OUT}${NC}"
else
    exec "${UVICORN_CMD[@]}"
fi
