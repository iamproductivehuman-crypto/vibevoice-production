#!/bin/bash
# =============================================================================
#  start_server.sh — Start the VibeVoice API server
#
#  Loads the model once, then serves HTTP requests on port 8080.
#  Keep this running in a tmux session so it survives SSH disconnects.
#
#  Usage:
#      bash start_server.sh               # foreground (Ctrl-C to stop)
#      bash start_server.sh --daemon      # background in current tmux pane
#
#  Tunnel from your local machine:
#      ssh -p PORT user@host -L 8080:localhost:8080
#      curl http://localhost:8080/health
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

HOST="${VIBEVOICE_HOST:-0.0.0.0}"
PORT="${VIBEVOICE_PORT:-8080}"
WORKERS=1          # always 1 — single GPU, model must not be forked
LOG_LEVEL="info"

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
python - <<'PY' 2>/dev/null || {
    echo "Installing server dependencies (fastapi, uvicorn)..."
    pip install "fastapi>=0.115" "uvicorn[standard]>=0.29" --quiet
}
import fastapi, uvicorn
PY

echo ""
echo "============================================================"
echo "  VibeVoice API Server"
echo "  http://${HOST}:${PORT}"
echo "  Docs: http://${HOST}:${PORT}/docs"
echo "  Health: http://${HOST}:${PORT}/health"
echo "============================================================"
echo ""

exec uvicorn server:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers "${WORKERS}" \
    --log-level "${LOG_LEVEL}" \
    --no-access-log
