"""
cleanup.py — Background WAV file cleanup for VibeVoice API
===========================================================
Runs as a background thread inside the FastAPI process.
Deletes files from OUTPUT_DIR that are older than CLEANUP_RETENTION_HOURS.

Configuration (via environment variables):
    CLEANUP_RETENTION_HOURS=24    # delete files older than this many hours (default: 24)
    CLEANUP_INTERVAL_S=3600       # how often to scan, in seconds (default: 3600 = 1 hour)
    VIBEVOICE_OUTPUT=./api_output # output directory (shared with server.py)
"""

import logging
import os
import threading
import time
from pathlib import Path

log = logging.getLogger("vibevoice-cleanup")

RETENTION_HOURS  = float(os.getenv("CLEANUP_RETENTION_HOURS", "24"))
INTERVAL_SECONDS = float(os.getenv("CLEANUP_INTERVAL_S",     "3600"))


def _cleanup_once(output_dir: Path) -> int:
    """
    Delete WAV files older than RETENTION_HOURS from output_dir.
    Returns the number of files deleted.
    """
    if not output_dir.is_dir():
        return 0

    cutoff = time.time() - RETENTION_HOURS * 3600
    deleted = 0
    for f in output_dir.glob("*.wav"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                log.info(f"Cleanup: deleted {f.name} (age > {RETENTION_HOURS}h)")
                deleted += 1
        except Exception as exc:
            log.warning(f"Cleanup: could not delete {f.name}: {exc}")
    return deleted


def _cleanup_loop(output_dir: Path) -> None:
    """Background loop — runs every INTERVAL_SECONDS."""
    log.info(
        f"Cleanup daemon started — retention={RETENTION_HOURS}h, "
        f"interval={INTERVAL_SECONDS}s, dir={output_dir}"
    )
    while True:
        try:
            n = _cleanup_once(output_dir)
            if n:
                log.info(f"Cleanup pass complete: {n} file(s) removed.")
            else:
                log.debug("Cleanup pass complete: nothing to remove.")
        except Exception as exc:
            log.error(f"Cleanup error: {exc}")
        time.sleep(INTERVAL_SECONDS)


def start_cleanup_daemon(output_dir: Path) -> threading.Thread:
    """
    Spawn a daemon thread that periodically deletes old output files.
    Safe to call from an asyncio context (pure thread, no event loop interaction).
    """
    t = threading.Thread(
        target=_cleanup_loop,
        args=(output_dir,),
        daemon=True,           # dies when main process exits
        name="vibevoice-cleanup",
    )
    t.start()
    return t
