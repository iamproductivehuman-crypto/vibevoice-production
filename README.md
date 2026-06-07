# VibeVoice Production

Self-hosted VibeVoice TTS API server with browser UI. One-command setup on a fresh Vast.ai GPU instance.

## Quick Start

```bash
git clone https://github.com/iamproductivehuman-crypto/vibevoice-production.git
cd vibevoice-production
bash install.sh --start-server
```

Takes ~10–15 minutes (mostly model download). When it finishes, the API is live and the browser UI is available.

## What install.sh does

| Step | Action |
|------|--------|
| 0 | GPU + CUDA check |
| 1 | Clone VibeVoice fork pinned to `4b9af0f` |
| 2 | Install PyTorch 2.5.1 + CUDA 12.1 |
| 3 | Install all Python dependencies |
| 4 | `pip install -e .` VibeVoice package |
| 5 | Apply patches (flash_attention_2 → sdpa) |
| 6 | Copy voice files from `voices/` |
| 7 | Download `microsoft/VibeVoice-1.5B` (~6 GB) |
| 8 | Validate install with test generation |
| 9 | Environment report |

At the end it prints:

```
Server running on:
  http://YOUR_IP:8000

Endpoints:
  GET  /health
  GET  /voices
  GET  /ui
  POST /generate
  POST /generate_url
  POST /batch_generate
  POST /upload_voice
```

## Browser UI

Open `http://YOUR_IP:8000/ui` in your browser.

**If `API_KEY` is set:** the UI shows a login screen. Enter your key once — it's stored in `localStorage`. All API requests automatically include it. No way to consume the GPU without it.

**Full workflow:**
1. Enter API key → unlocked and stored locally
2. Upload a folder of `.txt` scripts
3. Select a voice from dropdown (**Built-in Voices** / **Custom Voices** grouped separately)
4. **Test Voice** — type a sentence, click Preview, hear it before committing
5. Click **Generate All** → watch live per-script progress + queue indicator
6. Download individual WAVs or **Download ZIP** (all files in one click)
7. **History panel** at bottom — persists across page refreshes via `history.json`

## Voice Directories

| Directory | Purpose | Survives VibeVoice update? |
|-----------|---------|---|
| `voices/` | Committed to git; copied to `VibeVoice/demo/voices/` on install | No (re-copied on install) |
| `uploaded_voices/` | Custom voices uploaded via UI or copied manually | **Yes — completely separate** |

To add a voice without the UI:
```bash
cp narrator.wav vibevoice-production/uploaded_voices/
```

The UI shows both groups separately in the voice dropdown.

## Endpoints

### `GET /health`
Server status, GPU info, uptime, and queue depth. Always public (no API key required).

```bash
curl http://YOUR_IP:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "gpu_name": "NVIDIA RTX 4090",
  "vram_used_gb": 4.2,
  "vram_total_gb": 24.0,
  "vram_free_gb": 19.8,
  "uptime_s": 3600,
  "queue_length": 0
}
```

### `GET /voices`
List available voice stems.

```bash
curl http://YOUR_IP:8000/voices
```

### `POST /generate_url`
Generate speech from text, save to disk, return download URL.

```bash
curl -X POST http://YOUR_IP:8000/generate_url \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{
    "text": "Hello world, this is VibeVoice.",
    "voice": "en-Alice_woman",
    "cfg_scale": 1.32,
    "ddpm_steps": 10
  }'
```

```json
{
  "file_id": "abc123",
  "filename": "abc123.wav",
  "url": "/files/abc123.wav",
  "duration_s": 2.4,
  "generation_s": 3.1,
  "rtf": 1.29,
  "size_kb": 115.2
}
```

Download: `curl http://YOUR_IP:8000/files/abc123.wav -o output.wav`

### `POST /generate`
Generate speech and stream WAV bytes directly.

```bash
curl -X POST http://YOUR_IP:8000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"text": "Hello world", "voice": "en-Alice_woman"}' \
  -o output.wav
```

### `POST /batch_generate`
Generate multiple scripts in one call (up to 500 items).

```bash
curl -X POST http://YOUR_IP:8000/batch_generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{
    "items": [
      {"text": "First script.", "voice": "en-Alice_woman"},
      {"text": "Second script.", "voice": "en-Alice_woman"}
    ]
  }'
```

### `POST /upload_voice`
Upload a custom voice file (`.wav`, `.mp3`, `.flac`, `.ogg`).

```bash
curl -X POST http://YOUR_IP:8000/upload_voice \
  -H "X-API-Key: YOUR_KEY" \
  -F "file=@myvoice.wav"
```

## Python Client Example

```python
import requests

BASE = "http://YOUR_IP:8000"
HEADERS = {"X-API-Key": "YOUR_KEY"}  # omit if no key set

# Health check
r = requests.get(f"{BASE}/health", headers=HEADERS)
print(r.json())

# List voices
voices = requests.get(f"{BASE}/voices", headers=HEADERS).json()["voices"]
print("Available voices:", voices)

# Generate audio
response = requests.post(
    f"{BASE}/generate_url",
    headers=HEADERS,
    json={
        "text": "Welcome to VibeVoice. This is a test.",
        "voice": voices[0],
        "cfg_scale": 1.32,
        "ddpm_steps": 10,
    }
)
result = response.json()
print(f"Generated: {result['filename']} ({result['duration_s']:.1f}s)")

# Download the file
wav = requests.get(f"{BASE}{result['url']}")
with open("output.wav", "wb") as f:
    f.write(wav.content)
print("Saved to output.wav")
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | _(none — auth disabled)_ | If set, all generation endpoints require `X-API-Key: VALUE` |
| `PORT` | `8000` | Server listen port |
| `VIBEVOICE_MODEL` | `microsoft/VibeVoice-1.5B` | HuggingFace model ID |
| `VIBEVOICE_VOICES` | `VibeVoice/demo/voices` | Path to voice files |
| `VIBEVOICE_OUTPUT` | `api_output/` | Where generated WAVs are saved |
| `VIBEVOICE_DEFAULT_VOICE` | `en-Alice_woman` | Fallback voice |
| `VIBEVOICE_CFG_SCALE` | `1.32` | Default CFG scale |
| `VIBEVOICE_DDPM_STEPS` | `10` | Default DDPM steps |
| `CLEANUP_RETENTION_HOURS` | `24` | Delete output files older than this |
| `CLEANUP_INTERVAL_S` | `3600` | How often to scan for old files |

## Start Server (manual)

```bash
# Foreground (Ctrl-C to stop)
bash start_server.sh

# Background (nohup)
bash start_server.sh --daemon

# With API key
API_KEY=mysecretkey bash start_server.sh

# Custom port
PORT=9000 bash start_server.sh
```

## Add Voice Files

Drop `.wav`/`.mp3` files into `voices/` before running `install.sh`, or copy them after:

```bash
cp myvoice.wav vibevoice-production/VibeVoice/demo/voices/
```

Or use the browser UI (`/ui`) to upload voices without touching the filesystem.

## Text File Format

```
*title
Title of the video

*script
This is the script content.
Only text after *script is used for TTS.
```

Plain text (no markers) is also accepted.

## CLI Batch Generation

```bash
# List voices
python batch_generate.py --list_voices

# Convert all .txt files in a folder
python batch_generate.py \
  --input_dir  /path/to/txts \
  --output_dir /path/to/output \
  --speaker    Alice \
  --cfg_scale  1.32
```

Resume support: progress saved to `.batch_progress.json`. Re-run same command to skip already-done files.

## Patches Applied

| File | Fix |
|------|-----|
| `demo/inference_from_file.py` | `flash_attention_2` → `sdpa` |
| `demo/gradio_demo.py` | `flash_attention_2` → `sdpa`, Gradio API fixes |
| `demo/colab.py` | `flash_attention_2` → `sdpa` |

## Tips

- **RTF ~0.7x** on RTX 3060 — a 60s script generates in ~42s
- **OOM?** Script automatically skips and clears GPU cache
- **Model is cached** after first download at `~/.cache/huggingface/`
- **Vast.ai port 8080** is taken by Jupyter — this server uses **8000** by default
- **API docs**: `http://YOUR_IP:8000/docs` (Swagger UI)
- **Old WAVs** are automatically cleaned up after 24 hours (configurable)
