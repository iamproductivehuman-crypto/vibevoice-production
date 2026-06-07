#!/bin/bash
# =============================================================================
#  start_server.sh — Start the VibeVoice API server
#
#  Loads the model once, then serves HTTP requests on port 8000 by default.
#  Keep this running in a tmux session so it survives SSH disconnects.
#
#  Usage:
#      bash start_server.sh               # foreground (Ctrl-C to stop)
#      bash start_server.sh --daemon      # run in background via nohup
#
#  Environment variables:
#      PORT=8000                          # override listen port
#      VIBEVOICE_HOST=0.0.0.0            # override listen host
#      API_KEY=mysecretkey               # enable API key authentication
#      CLEANUP_RETENTION_HOURS=24        # hours to keep generated WAV files
#
#  Access from your local machine:
#      ssh -p PORT user@host -L 8000:localhost:8000
#      curl http://localhost:8000/health
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

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

# Check that FastAPI + uvicorn are installed (without heredoc+|| which breaks pipefail)
if ! python -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "Installing server dependencies (fastapi, uvicorn)..."
    pip install "fastapi>=0.115" "uvicorn[standard]>=0.29" --quiet
fi

# Resolve public IP for Vast.ai (or fall back to hostname)
SERVER_IP=$(curl -s --max-time 3 https://api.ipify.org 2>/dev/null \
    || curl -s --max-time 3 http://checkip.amazonaws.com 2>/dev/null \
    || hostname -I 2>/dev/null | awk '{print $1}' \
    || echo "localhost")

echo ""
echo "============================================================"
echo "  VibeVoice API Server"
echo "  Listening: http://${HOST}:${PORT}"
echo "  Public:    http://${SERVER_IP}:${PORT}"
echo ""
echo "  Endpoints:"
echo "    GET  http://${SERVER_IP}:${PORT}/health"
echo "    GET  http://${SERVER_IP}:${PORT}/voices"
echo "    GET  http://${SERVER_IP}:${PORT}/ui           (browser UI)"
echo "    POST http://${SERVER_IP}:${PORT}/generate"
echo "    POST http://${SERVER_IP}:${PORT}/generate_url"
echo "    POST http://${SERVER_IP}:${PORT}/batch_generate"
echo "    POST http://${SERVER_IP}:${PORT}/upload_voice"
echo "    GET  http://${SERVER_IP}:${PORT}/docs         (Swagger UI)"
echo "============================================================"
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
    echo "Starting in background. Logs: ${LOG_OUT}"
    nohup "${UVICORN_CMD[@]}" >> "${LOG_OUT}" 2>&1 &
    echo "Server PID: $!"
    echo "Stop with: kill $!"
else
    exec "${UVICORN_CMD[@]}"
fi
