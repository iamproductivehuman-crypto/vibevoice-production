"""
server.py — VibeVoice API Server
=================================
Loads the VibeVoice model ONCE at startup, keeps it resident in GPU
memory for the lifetime of the process, and serves HTTP generation
requests without ever reloading weights.

Start:
    uvicorn server:app --host 0.0.0.0 --port 8080 --workers 1

Or use the helper script:
    bash start_server.sh

Endpoints:
    GET  /health            liveness + GPU stats
    GET  /voices            list available voice files
    POST /generate          synthesise → stream WAV bytes back
    POST /generate_url      synthesise → save file → return JSON + URL
    POST /batch_generate    multiple texts in one call → JSON array of URLs
    GET  /files/{filename}  download a previously generated file
"""

import asyncio
import io
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
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

# Make the VibeVoice package importable
if str(VIBE_DIR) not in sys.path:
    sys.path.insert(0, str(VIBE_DIR))

MODEL_PATH        = os.getenv("VIBEVOICE_MODEL",   "microsoft/VibeVoice-1.5B")
VOICES_DIR        = Path(os.getenv("VIBEVOICE_VOICES",
                          str(VIBE_DIR / "demo" / "voices")))
OUTPUT_DIR        = Path(os.getenv("VIBEVOICE_OUTPUT",
                          str(SCRIPT_DIR / "api_output")))
DEFAULT_VOICE     = os.getenv("VIBEVOICE_DEFAULT_VOICE", "en-Alice_woman")
DEFAULT_CFG_SCALE = float(os.getenv("VIBEVOICE_CFG_SCALE",  "1.32"))
DEFAULT_DDPM_STEPS = int(os.getenv("VIBEVOICE_DDPM_STEPS",  "10"))
SAMPLE_RATE       = 24_000   # VibeVoice native output sample rate

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Global model state  — populated once in lifespan startup
# ---------------------------------------------------------------------------
_model     = None
_processor = None
# asyncio.Lock() serialises requests so only one inference runs at a time
# (single GPU — concurrency inside PyTorch is already maximised per request)
_lock: asyncio.Lock | None = None


# ---------------------------------------------------------------------------
# Lifespan: load model once, release cleanly on shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _processor, _lock

    _lock = asyncio.Lock()

    log.info("=" * 60)
    log.info("  VibeVoice API — starting up")
    log.info(f"  Model:      {MODEL_PATH}")
    log.info(f"  Voices dir: {VOICES_DIR}")
    log.info(f"  Output dir: {OUTPUT_DIR}")
    log.info("=" * 60)

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
        "One RTX 4090 can serve continuous generation requests."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Serve persisted WAV files for /generate_url and /batch_generate
app.mount("/files", StaticFiles(directory=str(OUTPUT_DIR)), name="files")


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
    items: list[BatchItem] = Field(..., min_length=1, max_length=100)


class BatchResultItem(BaseModel):
    index: int
    filename: str
    url: str
    duration_s: float
    generation_s: float
    size_kb: float
    error: Optional[str] = None


class BatchGenerateResponse(BaseModel):
    results: list[BatchResultItem]
    succeeded: int
    failed: int
    total_s: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_voice(voice_name: str) -> Path:
    """Return absolute path to a voice .wav file, or raise HTTP 404."""
    stem = voice_name.removesuffix(".wav")
    candidate = VOICES_DIR / f"{stem}.wav"
    if candidate.is_file():
        return candidate
    # Fuzzy match: accept partial / case-insensitive names
    sl = stem.lower()
    for p in VOICES_DIR.glob("*.wav"):
        if sl in p.stem.lower() or p.stem.lower() in sl:
            log.info(f"Voice fuzzy match: '{stem}' → '{p.stem}'")
            return p
    available = sorted(p.stem for p in VOICES_DIR.glob("*.wav"))
    raise HTTPException(
        status_code=404,
        detail=f"Voice '{stem}' not found. Available: {available}",
    )


def _format_script(text: str) -> str:
    """
    Accept plain text or *script-formatted text.
    Ensures lines are in 'Speaker 1: …' format that VibeVoice expects.
    Already-formatted text is passed through unchanged.
    """
    import re

    # Strip *title / *script markers (same logic as batch_generate.py)
    match = re.search(r'\*script\s*\n(.*)', text, re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1).strip()

    # If already in Speaker format, leave alone
    if re.match(r'^Speaker \d+:', text.strip()):
        return text.strip()

    # Convert plain paragraphs
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
    Run VibeVoice inference synchronously.
    Must be called while holding _lock (enforced by callers).
    Returns float32 numpy array of audio samples at SAMPLE_RATE Hz.
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
    # Move all tensors to GPU
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.cuda()

    # Override DDPM steps if caller requested something different from default
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

    # Reset to default steps for next request
    if ddpm_steps != DEFAULT_DDPM_STEPS:
        _model.set_ddpm_inference_steps(num_steps=DEFAULT_DDPM_STEPS)

    audio_tensor = outputs.speech_outputs[0]   # shape: (1, T) or (T,)
    return audio_tensor.squeeze().float().cpu().numpy()


def _audio_to_wav_bytes(audio_np: np.ndarray) -> bytes:
    """Encode numpy array → WAV bytes (in-memory, no disk write)."""
    buf = io.BytesIO()
    sf.write(buf, audio_np, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def _save_audio(audio_np: np.ndarray, filename: str) -> Path:
    """Write audio to OUTPUT_DIR and return the Path."""
    out_path = OUTPUT_DIR / filename
    sf.write(str(out_path), audio_np, SAMPLE_RATE, subtype="PCM_16")
    return out_path


# ---------------------------------------------------------------------------
# Routes — meta
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"], summary="Liveness + GPU stats")
async def health():
    if _model is None:
        raise HTTPException(503, "Model not loaded yet.")
    vram_used  = torch.cuda.memory_allocated() / 1e9
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return {
        "status":        "ok",
        "model":         MODEL_PATH,
        "gpu":           torch.cuda.get_device_name(0),
        "vram_used_gb":  round(vram_used,  2),
        "vram_total_gb": round(vram_total, 2),
        "voices_dir":    str(VOICES_DIR),
        "output_dir":    str(OUTPUT_DIR),
    }


@app.get("/voices", tags=["meta"], summary="List available voice files")
async def list_voices():
    voices = sorted(p.stem for p in VOICES_DIR.glob("*.wav"))
    if not voices:
        raise HTTPException(404, f"No voices found in {VOICES_DIR}")
    return {"voices": voices, "count": len(voices)}


# ---------------------------------------------------------------------------
# Routes — generation
# ---------------------------------------------------------------------------

@app.post(
    "/generate",
    tags=["generation"],
    summary="Synthesise text → stream WAV bytes back directly",
    response_class=StreamingResponse,
)
async def generate(req: GenerateRequest):
    """
    Generate speech and return the WAV file as a streaming response.
    Nothing is written to disk. The client receives a complete WAV file.
    """
    if _model is None:
        raise HTTPException(503, "Model not loaded.")

    voice_path = _resolve_voice(req.voice)
    t0 = time.time()

    async with _lock:
        log.info(
            f"/generate  voice={req.voice}  cfg={req.cfg_scale}  "
            f"steps={req.ddpm_steps}  "
            f"text={req.text[:70]!r}{'…' if len(req.text) > 70 else ''}"
        )
        try:
            audio_np = await asyncio.get_event_loop().run_in_executor(
                None,
                _run_inference,
                req.text, voice_path, req.cfg_scale, req.ddpm_steps,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise HTTPException(507, "CUDA out of memory. Try shorter text or restart server.")
        except Exception as exc:
            log.error(f"Inference error: {exc}", exc_info=True)
            raise HTTPException(500, f"Inference failed: {exc}")

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
)
async def generate_url(req: GenerateRequest):
    """
    Generate speech, save it to disk, and return a JSON body containing
    a /files/{filename} download URL. Useful for async or polling workflows.
    """
    if _model is None:
        raise HTTPException(503, "Model not loaded.")

    voice_path = _resolve_voice(req.voice)
    file_id    = uuid.uuid4().hex
    filename   = f"{file_id}.wav"
    t0         = time.time()

    async with _lock:
        log.info(f"/generate_url  voice={req.voice}  id={file_id}")
        try:
            audio_np = await asyncio.get_event_loop().run_in_executor(
                None,
                _run_inference,
                req.text, voice_path, req.cfg_scale, req.ddpm_steps,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise HTTPException(507, "CUDA out of memory.")
        except Exception as exc:
            log.error(f"Inference error: {exc}", exc_info=True)
            raise HTTPException(500, f"Inference failed: {exc}")

    out_path  = _save_audio(audio_np, filename)
    elapsed   = time.time() - t0
    duration  = len(audio_np) / SAMPLE_RATE
    size_kb   = round(out_path.stat().st_size / 1024, 1)
    log.info(f"  ✓ saved {filename}  dur={duration:.1f}s  gen={elapsed:.1f}s")

    return {
        "file_id":      file_id,
        "filename":     filename,
        "url":          f"/files/{filename}",
        "duration_s":   round(duration, 2),
        "generation_s": round(elapsed, 2),
        "rtf":          round(elapsed / max(duration, 0.01), 2),
        "size_kb":      size_kb,
    }


@app.post(
    "/batch_generate",
    tags=["generation"],
    summary="Synthesise multiple texts in one call → array of file URLs",
    response_model=BatchGenerateResponse,
)
async def batch_generate(req: BatchGenerateRequest):
    """
    Process a list of items sequentially (single GPU, one job at a time).
    All outputs are saved to OUTPUT_DIR. Response contains a URL for each.
    Failed items are included with error set rather than aborting the batch.
    """
    if _model is None:
        raise HTTPException(503, "Model not loaded.")

    results   = []
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
                duration_s=0, generation_s=0, size_kb=0,
                error=exc.detail,
            ))
            failed += 1
            continue

        t0 = time.time()
        async with _lock:
            log.info(
                f"/batch_generate [{idx+1}/{len(req.items)}]  "
                f"voice={item.voice}  text={item.text[:60]!r}"
            )
            try:
                audio_np = await asyncio.get_event_loop().run_in_executor(
                    None,
                    _run_inference,
                    item.text, voice_path, item.cfg_scale, item.ddpm_steps,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                err = "CUDA OOM — item skipped; cache cleared"
                log.error(f"  [{idx}] {err}")
                results.append(BatchResultItem(
                    index=idx, filename=filename, url="",
                    duration_s=0, generation_s=0, size_kb=0, error=err,
                ))
                failed += 1
                continue
            except Exception as exc:
                err = str(exc)
                log.error(f"  [{idx}] Inference error: {err}", exc_info=True)
                results.append(BatchResultItem(
                    index=idx, filename=filename, url="",
                    duration_s=0, generation_s=0, size_kb=0, error=err,
                ))
                failed += 1
                continue

        out_path  = _save_audio(audio_np, filename)
        elapsed   = time.time() - t0
        duration  = len(audio_np) / SAMPLE_RATE
        size_kb   = round(out_path.stat().st_size / 1024, 1)
        log.info(f"  ✓ {filename}  dur={duration:.1f}s  gen={elapsed:.1f}s")

        results.append(BatchResultItem(
            index=idx,
            filename=filename,
            url=f"/files/{filename}",
            duration_s=round(duration, 2),
            generation_s=round(elapsed, 2),
            size_kb=size_kb,
        ))
        succeeded += 1

    return BatchGenerateResponse(
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
    uvicorn.run("server:app", host="0.0.0.0", port=8080, workers=1, reload=False)
