# vibevoice-production

One-command VibeVoice setup for fresh Vast.ai GPU instances.

## First time on a new instance

```bash
git clone https://github.com/YOUR_USERNAME/vibevoice-production.git
cd vibevoice-production
bash install.sh
```

Takes ~10–15 minutes (mostly model download).

## Add your voice files

Drop your reference `.wav` or `.mp3` files into the `voices/` folder **before** running `install.sh`, or copy them manually afterward:

```bash
cp myvoice.wav /workspace/vibevoice-production/VibeVoice/demo/voices/
```

## Generate audio

```bash
# List available voices
python batch_generate.py --list_voices

# Convert all txt files
python batch_generate.py \
  --input_dir  /path/to/txt_files \
  --output_dir /path/to/output \
  --speaker    Alice \
  --cfg_scale  1.32

# Re-generate everything (ignore previous progress)
python batch_generate.py --input_dir ... --output_dir ... --reset
```

## Text file format

```
*title
Title of the video

*script
This is the script content.
Only text after *script is used for TTS.
```

## Resume support

Progress is saved to `.batch_progress.json`. If the instance dies mid-batch, just re-run the same command — already-converted files are skipped automatically.

## What install.sh does

1. Clones `microsoft/VibeVoice`
2. Installs all Python dependencies
3. Applies patches (`patch.py`): fixes `flash_attention_2 → sdpa` and Gradio compatibility issues
4. Downloads `microsoft/VibeVoice-1.5B` model weights (~6GB)
5. Copies your voices into `VibeVoice/demo/voices/`
6. Verifies GPU + voices

## Patches applied automatically

| File | Fix |
|---|---|
| `demo/inference_from_file.py` | `flash_attention_2` → `sdpa` |
| `demo/gradio_demo.py` | `flash_attention_2` → `sdpa`, Gradio API fixes |
| `demo/colab.py` | `flash_attention_2` → `sdpa` |

## Tips

- **RTF ~0.7x** on RTX 3060 — a 60s script generates in ~42s
- **OOM?** Script automatically skips and clears GPU cache
- **Model is cached** after first download at `~/.cache/huggingface/`
- **Speakers**: voice files are matched by name (partial, case-insensitive)
