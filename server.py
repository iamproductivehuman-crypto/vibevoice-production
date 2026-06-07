"""
server.py — VibeVoice API Server  v3.0
========================================
Loads the VibeVoice model ONCE at startup, keeps it resident in GPU
memory for the lifetime of the process, and serves HTTP generation
requests without ever reloading weights.

Start:
    uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1

Or use the helper script:
    bash start_server.sh [--daemon]

Environment variables:
    API_KEY                    — if set, all endpoints (except /health) require
                                 X-API-Key header with this value
    PORT                       — listen port (default 8000, used by __main__)
    VIBEVOICE_MODEL            — HuggingFace model ID (default: microsoft/VibeVoice-1.5B)
    VIBEVOICE_VOICES           — path to built-in voices dir (VibeVoice/demo/voices)
    VIBEVOICE_UPLOADED_VOICES  — path to custom uploaded voices (default: uploaded_voices/)
    VIBEVOICE_OUTPUT           — path to output directory (default: api_output/)
    VIBEVOICE_DEFAULT_VOICE    — default voice stem name
    VIBEVOICE_CFG_SCALE        — default CFG scale (default: 1.32)
    VIBEVOICE_DDPM_STEPS       — default DDPM steps (default: 10)
    CLEANUP_RETENTION_HOURS    — hours before output files are deleted (default: 24)
    CLEANUP_INTERVAL_S         — cleanup scan interval in seconds (default: 3600)
    HISTORY_MAX_ENTRIES        — max history entries to keep (default: 1000)

Endpoints:
    GET  /health                 liveness + GPU stats + queue info  [always public]
    GET  /voices                 list voices grouped: {builtin, custom}
    POST /upload_voice           upload .wav → uploaded_voices/
    POST /test_voice             quick preview generation (not saved to history)
    POST /generate               synthesise → stream WAV bytes back
    POST /generate_url           synthesise → save file → return JSON + URL
    POST /batch_generate         multiple texts → JSON array of URLs + job_id
    GET  /download_zip/{job_id}  download a ZIP of all files from a batch job
    GET  /history                generation history (persisted in history.json)
    GET  /files/{filename}       download a previously generated file
    GET  /ui                     browser UI (static/index.html)
"""

import asyncio
import io
import json
import logging
import os
import sys
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vibevoice-api")

# ---------------------------------------------------------------------------
# Paths & config  (all overridable via environment variables)
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent.resolve()
VIBE_DIR    = SCRIPT_DIR / "VibeVoice"
STATIC_DIR  = SCRIPT_DIR / "static"

# Make the VibeVoice package importable
if str(VIBE_DIR) not in sys.path:
    sys.path.insert(0, str(VIBE_DIR))

MODEL_PATH           = os.getenv("VIBEVOICE_MODEL",   "microsoft/VibeVoice-1.5B")
VOICES_DIR           = Path(os.getenv("VIBEVOICE_VOICES",
                             str(VIBE_DIR / "demo" / "voices")))     # built-in
UPLOADED_VOICES_DIR  = Path(os.getenv("VIBEVOICE_UPLOADED_VOICES",
                             str(SCRIPT_DIR / "uploaded_voices")))   # custom
OUTPUT_DIR           = Path(os.getenv("VIBEVOICE_OUTPUT",
                             str(SCRIPT_DIR / "api_output")))
DEFAULT_VOICE        = os.getenv("VIBEVOICE_DEFAULT_VOICE", "en-Alice_woman")
DEFAULT_CFG_SCALE    = float(os.getenv("VIBEVOICE_CFG_SCALE",  "1.32"))
DEFAULT_DDPM_STEPS   = int(os.getenv("VIBEVOICE_DDPM_STEPS",  "10"))
SAMPLE_RATE          = 24_000   # VibeVoice native output sample rate

HISTORY_FILE         = SCRIPT_DIR / "history.json"
HISTORY_MAX_ENTRIES  = int(os.getenv("HISTORY_MAX_ENTRIES", "1000"))

# API key authentication (optional — leave unset to disable)
API_KEY = os.getenv("API_KEY", "").strip() or None

# Create directories
for d in (OUTPUT_DIR, VOICES_DIR, UPLOADED_VOICES_DIR, STATIC_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Global model state  — populated once in lifespan startup
# ---------------------------------------------------------------------------
_model     = None
_processor = None
# asyncio.Lock() GUARANTEES only one inference at a time (single GPU).
# All other requests wait in the asyncio event loop — they never fail due
# to concurrent calls; they queue behind the lock.
_lock: asyncio.Lock | None = None
_queue_depth: int = 0          # requests waiting + currently running
_start_time: float = 0.0       # server start timestamp

# In-memory batch job store: job_id → list[filename]
# Used to serve /download_zip/{job_id}. Lost on server restart (files stay on disk).
_jobs: dict[str, list[str]] = {}


# ---------------------------------------------------------------------------
# API key dependency
# ---------------------------------------------------------------------------
async def require_api_key(x_api_key: str = Header(default="")) -> None:
    """FastAPI dependency — enforces API key if API_KEY env var is set."""
    if API_KEY is None:
        return   # auth disabled globally
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Set X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


# ---------------------------------------------------------------------------
# Lifespan: load model once, release cleanly on shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _processor, _lock, _start_time

    _lock = asyncio.Lock()
    _start_time = time.time()

    log.info("=" * 60)
    log.info("  VibeVoice API v3.0 — starting up")
    log.info(f"  Model:            {MODEL_PATH}")
    log.info(f"  Built-in voices:  {VOICES_DIR}")
    log.info(f"  Custom voices:    {UPLOADED_VOICES_DIR}")
    log.info(f"  Output dir:       {OUTPUT_DIR}")
    log.info(f"  Auth:             {'enabled' if API_KEY else 'disabled (no API_KEY set)'}")
    log.info("=" * 60)

    if API_KEY is None:
        log.warning("")
        log.warning("  ⚠  WARNING: API_KEY is not set.")
        log.warning("  ⚠  This server is PUBLICLY ACCESSIBLE — anyone who knows")
        log.warning("  ⚠  the URL can trigger GPU inference and consume your credits.")
        log.warning("  ⚠  Set API_KEY=your_secret_key before starting in production.")
        log.warning("")

    # Start background cleanup daemon
    from cleanup import start_cleanup_daemon
    start_cleanup_daemon(OUTPUT_DIR)

    t0 = time.time()
    try:
        from vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

        log.info("Loading processor…")
        _processor = VibeVoiceProcessor.from_pretrained(MODEL_PATH)

        log.info("Loading model into GPU…")
        _model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="sdpa",
        )
        _model.eval()
        _model.set_ddpm_inference_steps(num_steps=DEFAULT_DDPM_STEPS)

        vram_used  = torch.cuda.memory_allocated() / 1e9
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(
            f"Model ready in {time.time() - t0:.1f}s  "
            f"VRAM {vram_used:.1f}/{vram_total:.1f} GB  ✓"
        )
    except Exception as exc:
        log.error(f"Model load FAILED: {exc}")
        raise

    yield  # ← server is live here

    log.info("Shutdown — releasing GPU memory…")
    del _model, _processor
    torch.cuda.empty_cache()
    log.info("Done.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="VibeVoice API",
    description=(
        "Single-GPU TTS server. Model stays resident in VRAM. "
        "Requests queue automatically — never two simultaneous inferences."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

# Serve persisted WAV files
app.mount("/files", StaticFiles(directory=str(OUTPUT_DIR)), name="files")

# Serve browser UI from static/
if STATIC_DIR.is_dir() and any(STATIC_DIR.iterdir()):
    app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    text: str = Field(
        ...,
        description=(
            "Text to synthesise. "
            "Use 'Speaker 1: line' format or plain text — "
            "the server will format it automatically."
        ),
    )
    voice: str = Field(
        DEFAULT_VOICE,
        description="Voice stem name, e.g. 'en-Alice_woman'. Extension optional.",
    )
    cfg_scale: float  = Field(DEFAULT_CFG_SCALE,  ge=0.5, le=3.0)
    ddpm_steps: int   = Field(DEFAULT_DDPM_STEPS, ge=1,   le=50)


class BatchItem(BaseModel):
    text: str
    voice: str          = DEFAULT_VOICE
    cfg_scale: float    = DEFAULT_CFG_SCALE
    ddpm_steps: int     = DEFAULT_DDPM_STEPS
    output_filename: Optional[str] = None   # auto-generated if omitted


class BatchGenerateRequest(BaseModel):
    items: list[BatchItem] = Field(..., min_length=1, max_length=500)


class BatchResultItem(BaseModel):
    index: int
    filename: str
    url: str
    duration_s: float
    generation_s: float
    size_kb: float
    error: Optional[str] = None


class BatchGenerateResponse(BaseModel):
    job_id: str
    download_zip_url: str
    results: list[BatchResultItem]
    succeeded: int
    failed: int
    total_s: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_voices_grouped() -> dict:
    """Return voices split into built-in and custom groups."""
    builtin = sorted(p.stem for p in VOICES_DIR.glob("*.wav"))
    custom  = sorted(p.stem for p in UPLOADED_VOICES_DIR.glob("*.wav"))
    return {
        "builtin": builtin,
        "custom":  custom,
        "count":   len(builtin) + len(custom),
    }


def _resolve_voice(voice_name: str) -> Path:
    """
    Return absolute path to a voice .wav file, or raise HTTP 404.
    Custom (uploaded) voices take priority over built-in voices.
    """
    stem = voice_name.removesuffix(".wav")

    # Exact match — custom first, then built-in
    for vdir in (UPLOADED_VOICES_DIR, VOICES_DIR):
        candidate = vdir / f"{stem}.wav"
        if candidate.is_file():
            return candidate

    # Fuzzy match — custom first, then built-in
    sl = stem.lower()
    for vdir in (UPLOADED_VOICES_DIR, VOICES_DIR):
        for p in vdir.glob("*.wav"):
            if sl in p.stem.lower() or p.stem.lower() in sl:
                log.info(f"Voice fuzzy match: '{stem}' → '{p.stem}' in {vdir.name}/")
                return p

    available = (
        [f"[custom] {p.stem}"  for p in UPLOADED_VOICES_DIR.glob("*.wav")] +
        [f"[builtin] {p.stem}" for p in VOICES_DIR.glob("*.wav")]
    )
    raise HTTPException(
        status_code=404,
        detail=f"Voice '{stem}' not found. Available: {sorted(available)}",
    )


def _format_script(text: str) -> str:
    """
    Accept plain text or *script-formatted text.
    Ensures lines are in 'Speaker 1: …' format that VibeVoice expects.
    """
    import re

    match = re.search(r'\*script\s*\n(.*)', text, re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1).strip()

    if re.match(r'^Speaker \d+:', text.strip()):
        return text.strip()

    lines = [p.strip() for p in re.split(r'\n\n+|\n', text) if p.strip()]
    if not lines:
        lines = [text.strip()]
    return "\n".join(f"Speaker 1: {line}" for line in lines)


def _load_voice_audio(voice_path: Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load a voice reference file to a float32 mono numpy array."""
    import librosa
    wav, orig_sr = sf.read(str(voice_path))
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if orig_sr != sr:
        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=sr)
    return wav.astype(np.float32)


def _run_inference(
    text: str,
    voice_path: Path,
    cfg_scale: float,
    ddpm_steps: int,
) -> np.ndarray:
    """
    Run VibeVoice inference synchronously (blocking).
    MUST be called while the caller holds _lock.
    Returns float32 numpy array of audio at SAMPLE_RATE Hz.
    """
    voice_audio = _load_voice_audio(voice_path)
    script      = _format_script(text)

    inputs = _processor(
        text=[script],
        voice_samples=[[voice_audio]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.cuda()

    if ddpm_steps != DEFAULT_DDPM_STEPS:
        _model.set_ddpm_inference_steps(num_steps=ddpm_steps)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=cfg_scale,
            tokenizer=_processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
        )

    if ddpm_steps != DEFAULT_DDPM_STEPS:
        _model.set_ddpm_inference_steps(num_steps=DEFAULT_DDPM_STEPS)

    audio_tensor = outputs.speech_outputs[0]
    return audio_tensor.squeeze().float().cpu().numpy()


def _audio_to_wav_bytes(audio_np: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio_np, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def _save_audio(audio_np: np.ndarray, filename: str) -> Path:
    out_path = OUTPUT_DIR / filename
    sf.write(str(out_path), audio_np, SAMPLE_RATE, subtype="PCM_16")
    return out_path


def _append_history(entry: dict) -> None:
    """Append one entry to history.json, capping at HISTORY_MAX_ENTRIES."""
    try:
        history: list = json.loads(HISTORY_FILE.read_text()) if HISTORY_FILE.is_file() else []
        history.append(entry)
        if len(history) > HISTORY_MAX_ENTRIES:
            history = history[-HISTORY_MAX_ENTRIES:]
        HISTORY_FILE.write_text(json.dumps(history, indent=2))
    except Exception as exc:
        log.warning(f"Could not write history.json: {exc}")


async def _do_inference(text: str, voice_path: Path, cfg_scale: float, ddpm_steps: int) -> np.ndarray:
    """Acquire the GPU lock, run inference, release lock. Handles OOM."""
    global _queue_depth
    _queue_depth += 1
    try:
        async with _lock:
            try:
                return await asyncio.get_event_loop().run_in_executor(
                    None, _run_inference, text, voice_path, cfg_scale, ddpm_steps,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise HTTPException(507, "CUDA out of memory. Try shorter text or restart server.")
            except Exception as exc:
                log.error(f"Inference error: {exc}", exc_info=True)
                raise HTTPException(500, f"Inference failed: {exc}")
    finally:
        _queue_depth -= 1


# ---------------------------------------------------------------------------
# Routes — meta (always public or lightly gated)
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"], summary="Liveness + GPU stats + queue info")
async def health():
    """
    Always public — no API key required.
    Returns model status, GPU info, uptime, and current queue depth.
    """
    uptime_s = round(time.time() - _start_time, 1) if _start_time else 0

    base = {
        "status":        "ok",
        "model_loaded":  _model is not None,
        "uptime_s":      uptime_s,
        "queue_length":  _queue_depth,
        "auth_enabled":  API_KEY is not None,
    }

    if not torch.cuda.is_available():
        return {**base, "gpu_name": None, "vram_used_gb": None, "vram_total_gb": None}

    vram_used  = torch.cuda.memory_allocated() / 1e9
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return {
        **base,
        "model":         MODEL_PATH,
        "gpu_name":      torch.cuda.get_device_name(0),
        "vram_used_gb":  round(vram_used,  2),
        "vram_total_gb": round(vram_total, 2),
        "vram_free_gb":  round(vram_total - vram_used, 2),
    }


@app.get("/", tags=["meta"], summary="Status homepage", response_class=HTMLResponse)
async def homepage():
    """
    Public homepage — shows server status, GPU info, and access instructions.
    No API key required. Fetches live /health data via JS.
    """
    auth_note = (
        '<div class="warn-banner">⚠️ <strong>Auth is disabled.</strong> '
        'Set the <code>API_KEY</code> environment variable to protect this server.</div>'
        if API_KEY is None
        else '<div class="ok-banner">🔐 API key authentication is enabled.</div>'
    )
    return HTMLResponse(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>VibeVoice Server</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#090a0d;--bg2:rgba(255,255,255,.035);--border:rgba(255,255,255,.08);
  --accent:#8b5cf6;--accent2:#06b6d4;--txt:#f1f5f9;--txt2:#94a3b8;--txt3:#475569;
  --ok:#10b981;--warn:#f59e0b;--err:#ef4444;
  --grad:linear-gradient(135deg,#8b5cf6,#06b6d4);
}}
body{{font-family:Inter,system-ui,sans-serif;background:var(--bg);color:var(--txt);
  min-height:100vh;line-height:1.6;-webkit-font-smoothing:antialiased}}
body::before{{content:'';position:fixed;top:-25%;left:-8%;width:500px;height:500px;
  border-radius:50%;background:radial-gradient(circle,rgba(139,92,246,.1) 0%,transparent 70%);
  pointer-events:none}}
.shell{{max-width:820px;margin:0 auto;padding:48px 20px 80px}}
.hero{{text-align:center;margin-bottom:48px}}
.logo-ring{{width:68px;height:68px;border-radius:20px;background:var(--grad);
  display:flex;align-items:center;justify-content:center;font-size:32px;
  margin:0 auto 16px;box-shadow:0 8px 32px rgba(139,92,246,.35)}}
h1{{font-size:2rem;font-weight:700;background:var(--grad);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  margin-bottom:6px}}
.sub{{color:var(--txt2);font-size:.9rem}}
.warn-banner{{background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.35);
  border-radius:10px;padding:12px 16px;font-size:.85rem;margin-bottom:24px;color:#fcd34d}}
.ok-banner{{background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.3);
  border-radius:10px;padding:12px 16px;font-size:.85rem;margin-bottom:24px;color:#6ee7b7}}
.card{{background:var(--bg2);border:1px solid var(--border);border-radius:14px;
  padding:22px;margin-bottom:18px;backdrop-filter:blur(12px)}}
.card-title{{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;
  color:var(--txt2);margin-bottom:14px}}
.stat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px}}
.stat{{text-align:center;padding:14px 10px;background:rgba(255,255,255,.025);
  border-radius:10px;border:1px solid var(--border)}}
.sv{{font-size:1.3rem;font-weight:700;background:var(--grad);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}}
.sl{{font-size:.65rem;color:var(--txt3);text-transform:uppercase;letter-spacing:.06em;margin-top:2px}}
.btn-row{{display:flex;flex-wrap:wrap;gap:10px;margin-top:16px}}
.btn{{display:inline-flex;align-items:center;gap:7px;font-family:inherit;
  font-weight:600;font-size:.9rem;border:none;border-radius:9px;padding:11px 20px;
  cursor:pointer;text-decoration:none;transition:opacity .15s,box-shadow .2s}}
.btn-primary{{background:var(--grad);color:#fff;box-shadow:0 4px 18px rgba(139,92,246,.3)}}
.btn-primary:hover{{opacity:.85;box-shadow:0 6px 26px rgba(139,92,246,.45)}}
.btn-ghost{{background:rgba(255,255,255,.05);border:1px solid var(--border);color:var(--txt2)}}
.btn-ghost:hover{{background:rgba(255,255,255,.09);color:var(--txt)}}
pre{{background:rgba(0,0,0,.4);border:1px solid var(--border);border-radius:9px;
  padding:14px 16px;font-size:.82rem;color:var(--txt2);overflow-x:auto;
  font-family:'JetBrains Mono',Menlo,monospace;line-height:1.7}}
code{{background:rgba(255,255,255,.07);border-radius:4px;padding:1px 5px;font-size:.82rem}}
.ep-list{{list-style:none;display:flex;flex-direction:column;gap:5px}}
.ep-list li{{display:flex;gap:10px;align-items:center;font-size:.82rem}}
.method{{font-weight:700;font-size:.7rem;padding:2px 7px;border-radius:4px;min-width:46px;text-align:center}}
.get{{background:rgba(16,185,129,.15);color:#6ee7b7}}
.post{{background:rgba(139,92,246,.15);color:#c4b5fd}}
.ep-path{{color:var(--txt);font-family:Menlo,monospace;font-size:.8rem}}
.ep-desc{{color:var(--txt3);margin-left:auto;font-size:.75rem}}
.dot{{width:9px;height:9px;border-radius:50%;background:var(--txt3);display:inline-block}}
.dot.online{{background:var(--ok);box-shadow:0 0 8px var(--ok)}}
.dot.loading{{background:var(--warn)}}
.dot.err{{background:var(--err)}}
@media(max-width:600px){{.ep-desc{{display:none}}}}
</style>
</head>
<body>
<div class="shell">
  <div class="hero">
    <div class="logo-ring">🎙️</div>
    <h1>VibeVoice Server</h1>
    <p class="sub">Self-hosted GPU TTS · Model resident in VRAM · Requests queued automatically</p>
  </div>

  {auth_note}

  <!-- Live status card -->
  <div class="card">
    <div class="card-title">⚡ Server Status</div>
    <div class="stat-grid" id="stats">
      <div class="stat"><div class="sv" id="s-status"><span class="dot loading"></span></div><div class="sl">Status</div></div>
      <div class="stat"><div class="sv" id="s-gpu">…</div><div class="sl">GPU</div></div>
      <div class="stat"><div class="sv" id="s-vram">…</div><div class="sl">VRAM Used</div></div>
      <div class="stat"><div class="sv" id="s-uptime">…</div><div class="sl">Uptime</div></div>
      <div class="stat"><div class="sv" id="s-queue">…</div><div class="sl">Queue</div></div>
    </div>
    <div class="btn-row">
      <a class="btn btn-primary" href="/ui">🎛️ Open Studio UI</a>
      <a class="btn btn-ghost" href="/docs">📖 API Docs (Swagger)</a>
      <a class="btn btn-ghost" href="/health">🩺 Health JSON</a>
    </div>
  </div>

  <!-- SSH access card -->
  <div class="card">
    <div class="card-title">🔗 Remote Access (Vast.ai / SSH Tunneling)</div>
    <p style="font-size:.82rem;color:var(--txt2);margin-bottom:12px">
      Vast.ai does not automatically expose port 8000. Use an SSH tunnel from your local machine:
    </p>
    <pre id="ssh-cmd">ssh -p SSH_PORT root@SERVER_IP -L 8000:localhost:8000</pre>
    <p style="font-size:.75rem;color:var(--txt3);margin-top:8px">
      Then open <code>http://localhost:8000/ui</code> in your browser.
    </p>
    <p style="font-size:.75rem;color:var(--txt3);margin-top:4px">
      Find your SSH port in the Vast.ai dashboard under your instance's connection details.
    </p>
  </div>

  <!-- Endpoints card -->
  <div class="card">
    <div class="card-title">📡 API Endpoints</div>
    <ul class="ep-list">
      <li><span class="method get">GET</span><span class="ep-path">/health</span><span class="ep-desc">Liveness + GPU stats (public)</span></li>
      <li><span class="method get">GET</span><span class="ep-path">/voices</span><span class="ep-desc">List built-in + custom voices</span></li>
      <li><span class="method post">POST</span><span class="ep-path">/generate</span><span class="ep-desc">Stream WAV bytes</span></li>
      <li><span class="method post">POST</span><span class="ep-path">/generate_url</span><span class="ep-desc">Save file → return URL</span></li>
      <li><span class="method post">POST</span><span class="ep-path">/batch_generate</span><span class="ep-desc">Multiple texts → URLs + ZIP</span></li>
      <li><span class="method post">POST</span><span class="ep-path">/test_voice</span><span class="ep-desc">Quick preview (not in history)</span></li>
      <li><span class="method post">POST</span><span class="ep-path">/upload_voice</span><span class="ep-desc">Upload voice to uploaded_voices/</span></li>
      <li><span class="method get">GET</span><span class="ep-path">/download_zip/{{job_id}}</span><span class="ep-desc">ZIP from batch job</span></li>
      <li><span class="method get">GET</span><span class="ep-path">/history</span><span class="ep-desc">Generation history (JSON)</span></li>
      <li><span class="method get">GET</span><span class="ep-path">/ui</span><span class="ep-desc">Browser UI</span></li>
    </ul>
  </div>
</div>

<script>
async function fetchStatus() {{
  try {{
    const h = await fetch('/health').then(r=>r.json());
    const model = h.model_loaded ? '✅ Ready' : '⏳ Loading';
    document.getElementById('s-status').innerHTML =
      `<span class="dot ${{h.model_loaded?'online':'loading'}}"></span> ${{model}}`;
    document.getElementById('s-gpu').textContent    = h.gpu_name || 'N/A';
    document.getElementById('s-vram').textContent   = h.vram_used_gb!=null ? `${{h.vram_used_gb}}/${{h.vram_total_gb}} GB` : 'N/A';
    const u=h.uptime_s||0;
    document.getElementById('s-uptime').textContent = u<60 ? u+'s' : u<3600 ? Math.floor(u/60)+'m' : Math.floor(u/3600)+'h '+Math.floor((u%3600)/60)+'m';
    document.getElementById('s-queue').textContent  = h.queue_length ?? 0;
  }} catch {{
    document.getElementById('s-status').innerHTML = '<span class="dot err"></span> Offline';
  }}
}}
fetchStatus();
setInterval(fetchStatus, 10000);
</script>
</body>
</html>""")


@app.get(
    "/voices",
    tags=["meta"],
    summary="List voices grouped into built-in and custom",
    dependencies=[Depends(require_api_key)],
)
async def list_voices():
    """Returns {builtin: [...], custom: [...], count: N}."""
    return _get_voices_grouped()


@app.post(
    "/upload_voice",
    tags=["meta"],
    summary="Upload a custom voice file → saved to uploaded_voices/",
    dependencies=[Depends(require_api_key)],
)
async def upload_voice(file: UploadFile = File(...)):
    """
    Upload a .wav/.mp3/.flac/.ogg file.
    Saved to uploaded_voices/ (separate from VibeVoice built-ins).
    The new voice is immediately available for generation.
    """
    if not file.filename:
        raise HTTPException(400, "No filename provided.")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".wav", ".mp3", ".flac", ".ogg"}:
        raise HTTPException(400, f"Unsupported format: {suffix}. Use .wav/.mp3/.flac/.ogg")

    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in file.filename)
    dest = UPLOADED_VOICES_DIR / safe_name
    content = await file.read()
    dest.write_bytes(content)
    log.info(f"Voice uploaded: {safe_name} ({len(content) // 1024} KB) → uploaded_voices/")

    return {
        "uploaded":  safe_name,
        "stem":      Path(safe_name).stem,
        "size_kb":   round(len(content) / 1024, 1),
        "voices":    _get_voices_grouped(),
    }


@app.get(
    "/history",
    tags=["meta"],
    summary="Generation history (persisted across page refreshes)",
    dependencies=[Depends(require_api_key)],
)
async def get_history(limit: int = 200):
    """Returns the last `limit` generation entries from history.json."""
    if not HISTORY_FILE.is_file():
        return {"entries": [], "count": 0}
    try:
        history: list = json.loads(HISTORY_FILE.read_text())
        entries = history[-limit:] if len(history) > limit else history
        return {"entries": list(reversed(entries)), "count": len(history)}
    except Exception as exc:
        raise HTTPException(500, f"Could not read history: {exc}")


@app.get(
    "/download_zip/{job_id}",
    tags=["generation"],
    summary="Download all files from a batch job as a ZIP",
    dependencies=[Depends(require_api_key)],
)
async def download_zip(job_id: str):
    """
    Server-side ZIP: collects all WAV files from a batch_generate job
    and streams a ZIP archive back to the client.
    """
    if job_id not in _jobs:
        raise HTTPException(404, f"Job '{job_id}' not found. Jobs are lost on server restart.")

    filenames = _jobs[job_id]
    # Filter to files that actually exist
    paths = [(fn, OUTPUT_DIR / fn) for fn in filenames if (OUTPUT_DIR / fn).is_file()]
    if not paths:
        raise HTTPException(410, "All files from this job have been cleaned up.")

    def generate_zip():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            for fn, path in paths:
                zf.write(path, arcname=fn)
        buf.seek(0)
        yield buf.read()

    return StreamingResponse(
        generate_zip(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="vibevoice_{job_id[:8]}.zip"',
        },
    )


# ---------------------------------------------------------------------------
# Routes — generation
# ---------------------------------------------------------------------------

@app.post(
    "/test_voice",
    tags=["generation"],
    summary="Quick preview generation — not saved to history",
    dependencies=[Depends(require_api_key)],
)
async def test_voice(req: GenerateRequest):
    """
    Generate a short voice preview. Same as /generate_url but:
    - Not recorded in history.json
    - File gets a 'test_' prefix for easy identification
    Intended for trying voices before committing to a full batch.
    """
    if _model is None:
        raise HTTPException(503, "Model not loaded.")

    voice_path = _resolve_voice(req.voice)
    filename   = f"test_{uuid.uuid4().hex[:8]}.wav"
    t0         = time.time()

    log.info(f"/test_voice  voice={req.voice}  text={req.text[:60]!r}")
    audio_np  = await _do_inference(req.text, voice_path, req.cfg_scale, req.ddpm_steps)
    out_path  = _save_audio(audio_np, filename)
    elapsed   = time.time() - t0
    duration  = len(audio_np) / SAMPLE_RATE
    log.info(f"  ✓ test preview: {filename}  dur={duration:.1f}s  gen={elapsed:.1f}s")

    return {
        "filename":     filename,
        "url":          f"/files/{filename}",
        "duration_s":   round(duration, 2),
        "generation_s": round(elapsed, 2),
        "rtf":          round(elapsed / max(duration, 0.01), 2),
        "is_test":      True,
    }


@app.post(
    "/generate",
    tags=["generation"],
    summary="Synthesise text → stream WAV bytes back directly",
    response_class=StreamingResponse,
    dependencies=[Depends(require_api_key)],
)
async def generate(req: GenerateRequest):
    """
    Generate speech and return the WAV file as a streaming response.
    Nothing is written to disk.
    Requests queue automatically — they wait behind the GPU lock rather than fail.
    """
    if _model is None:
        raise HTTPException(503, "Model not loaded.")

    voice_path = _resolve_voice(req.voice)
    t0 = time.time()

    log.info(
        f"/generate  voice={req.voice}  cfg={req.cfg_scale}  "
        f"steps={req.ddpm_steps}  text={req.text[:70]!r}"
    )
    audio_np  = await _do_inference(req.text, voice_path, req.cfg_scale, req.ddpm_steps)
    wav_bytes = _audio_to_wav_bytes(audio_np)
    elapsed   = time.time() - t0
    duration  = len(audio_np) / SAMPLE_RATE
    log.info(
        f"  ✓ dur={duration:.1f}s  gen={elapsed:.1f}s  "
        f"RTF={elapsed/max(duration,0.01):.2f}x  {len(wav_bytes)//1024}KB"
    )

    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={
            "Content-Disposition":  'attachment; filename="output.wav"',
            "Content-Length":       str(len(wav_bytes)),
            "X-Duration-S":         str(round(duration, 2)),
            "X-Generation-S":       str(round(elapsed, 2)),
            "X-RTF":                str(round(elapsed / max(duration, 0.01), 2)),
        },
    )


@app.post(
    "/generate_url",
    tags=["generation"],
    summary="Synthesise text → save file → return JSON with download URL",
    dependencies=[Depends(require_api_key)],
)
async def generate_url(req: GenerateRequest):
    """
    Generate speech, save to disk, return JSON with /files/{filename} URL.
    Generation is appended to history.json.
    Requests queue automatically.
    """
    if _model is None:
        raise HTTPException(503, "Model not loaded.")

    voice_path = _resolve_voice(req.voice)
    file_id    = uuid.uuid4().hex
    filename   = f"{file_id}.wav"
    t0         = time.time()

    log.info(f"/generate_url  voice={req.voice}  id={file_id}")
    audio_np  = await _do_inference(req.text, voice_path, req.cfg_scale, req.ddpm_steps)
    out_path  = _save_audio(audio_np, filename)
    elapsed   = time.time() - t0
    duration  = len(audio_np) / SAMPLE_RATE
    size_kb   = round(out_path.stat().st_size / 1024, 1)
    log.info(f"  ✓ saved {filename}  dur={duration:.1f}s  gen={elapsed:.1f}s")

    result = {
        "file_id":      file_id,
        "filename":     filename,
        "url":          f"/files/{filename}",
        "voice":        req.voice,
        "duration_s":   round(duration, 2),
        "generation_s": round(elapsed, 2),
        "rtf":          round(elapsed / max(duration, 0.01), 2),
        "size_kb":      size_kb,
    }

    _append_history({
        **result,
        "text_preview": req.text[:120],
        "timestamp":    int(time.time()),
        "job_id":       None,
    })

    return result


@app.post(
    "/batch_generate",
    tags=["generation"],
    summary="Synthesise multiple texts → array of URLs + downloadable ZIP",
    response_model=BatchGenerateResponse,
    dependencies=[Depends(require_api_key)],
)
async def batch_generate(req: BatchGenerateRequest):
    """
    Process a list of items sequentially (single GPU lock — never concurrent).
    All outputs are saved to OUTPUT_DIR.
    Returns a job_id usable with GET /download_zip/{job_id}.
    Failed items include error rather than aborting the entire batch.
    Supports up to 500 items per call.
    """
    if _model is None:
        raise HTTPException(503, "Model not loaded.")

    job_id    = uuid.uuid4().hex
    results   = []
    filenames = []   # track for ZIP
    total_t0  = time.time()
    succeeded = 0
    failed    = 0

    for idx, item in enumerate(req.items):
        filename = item.output_filename or f"{uuid.uuid4().hex}.wav"
        if not filename.endswith(".wav"):
            filename += ".wav"

        try:
            voice_path = _resolve_voice(item.voice)
        except HTTPException as exc:
            results.append(BatchResultItem(
                index=idx, filename=filename, url="",
                duration_s=0, generation_s=0, size_kb=0, error=exc.detail,
            ))
            failed += 1
            continue

        t0 = time.time()
        log.info(
            f"/batch_generate [{idx+1}/{len(req.items)}] job={job_id[:8]}  "
            f"voice={item.voice}  text={item.text[:60]!r}"
        )
        try:
            audio_np = await _do_inference(
                item.text, voice_path, item.cfg_scale, item.ddpm_steps
            )
        except HTTPException as exc:
            results.append(BatchResultItem(
                index=idx, filename=filename, url="",
                duration_s=0, generation_s=0, size_kb=0, error=exc.detail,
            ))
            failed += 1
            continue

        out_path  = _save_audio(audio_np, filename)
        elapsed   = time.time() - t0
        duration  = len(audio_np) / SAMPLE_RATE
        size_kb   = round(out_path.stat().st_size / 1024, 1)
        log.info(f"  ✓ {filename}  dur={duration:.1f}s  gen={elapsed:.1f}s")

        filenames.append(filename)
        results.append(BatchResultItem(
            index=idx,
            filename=filename,
            url=f"/files/{filename}",
            duration_s=round(duration, 2),
            generation_s=round(elapsed, 2),
            size_kb=size_kb,
        ))
        succeeded += 1

        _append_history({
            "file_id":      filename.removesuffix(".wav"),
            "filename":     filename,
            "url":          f"/files/{filename}",
            "voice":        item.voice,
            "duration_s":   round(duration, 2),
            "generation_s": round(elapsed, 2),
            "rtf":          round(elapsed / max(duration, 0.01), 2),
            "size_kb":      size_kb,
            "text_preview": item.text[:120],
            "timestamp":    int(time.time()),
            "job_id":       job_id,
        })

    # Store job for ZIP download
    _jobs[job_id] = filenames

    return BatchGenerateResponse(
        job_id=job_id,
        download_zip_url=f"/download_zip/{job_id}",
        results=results,
        succeeded=succeeded,
        failed=failed,
        total_s=round(time.time() - total_t0, 2),
    )


# ---------------------------------------------------------------------------
# Dev entry-point  (use uvicorn directly in production)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, workers=1, reload=False)
