"""
batch_generate.py
─────────────────
Batch-convert a folder of .txt files to audio using VibeVoice.

Text file format expected:
    *title
    Title of the video
    *script
    Script content here...

Only the text after *script is used.

Usage:
    python batch_generate.py \\
        --input_dir  /path/to/txts \\
        --output_dir /path/to/output \\
        --speaker    Alice \\
        --cfg_scale  1.32 \\
        --model_path microsoft/VibeVoice-1.5B
"""

import argparse
import os
import re
import sys
import time
import json
import torch
import soundfile as sf
import numpy as np

# ── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
VIBE_DIR    = os.path.join(SCRIPT_DIR, "VibeVoice")
VOICES_DIR  = os.path.join(VIBE_DIR, "demo", "voices")
PROGRESS_FILE = os.path.join(SCRIPT_DIR, ".batch_progress.json")

if VIBE_DIR not in sys.path:
    sys.path.insert(0, VIBE_DIR)

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
except ImportError:
    print("ERROR: VibeVoice not found. Run 'bash install.sh' first.")
    sys.exit(1)

import librosa

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_script(text):
    """Return only text after *script marker."""
    match = re.search(r'\*script\s*\n(.*)', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def format_script(text, speaker="Speaker 1"):
    """Format plain text into VibeVoice Speaker format."""
    lines = [p.strip() for p in re.split(r'\n\n+|\n', text) if p.strip()]
    if not lines:
        lines = [text.strip()]
    return "\n".join(f"{speaker}: {l}" for l in lines)

def get_voice_path(speaker_name):
    """Find voice file for a speaker name (same logic as VoiceMapper)."""
    if not os.path.isdir(VOICES_DIR):
        raise FileNotFoundError(f"Voices directory not found: {VOICES_DIR}")

    files = [f for f in os.listdir(VOICES_DIR)
             if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]

    # Build name → path dict (same cleanup as VoiceMapper)
    presets = {}
    for f in files:
        name = os.path.splitext(f)[0]
        presets[name] = os.path.join(VOICES_DIR, f)
        if '_' in name:
            presets[name.split('_')[0]] = os.path.join(VOICES_DIR, f)
        if '-' in name:
            presets[name.split('-')[-1]] = os.path.join(VOICES_DIR, f)

    # Exact match
    if speaker_name in presets:
        return presets[speaker_name]

    # Case-insensitive partial match
    sl = speaker_name.lower()
    for k, v in presets.items():
        if sl in k.lower() or k.lower() in sl:
            return v

    # Default to first
    first = os.path.join(VOICES_DIR, files[0])
    print(f"  Warning: speaker '{speaker_name}' not found, using {files[0]}")
    return first

def load_audio(path, sr=24000):
    wav, orig_sr = sf.read(path)
    if len(wav.shape) > 1:
        wav = np.mean(wav, axis=1)
    if orig_sr != sr:
        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=sr)
    return wav

def load_progress():
    if os.path.isfile(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return set(json.load(f))
    return set()

def save_progress(done):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(list(done), f)

def list_voices():
    if not os.path.isdir(VOICES_DIR):
        print(f"Voices directory not found: {VOICES_DIR}")
        return
    files = [f for f in os.listdir(VOICES_DIR)
             if f.lower().endswith(('.wav','.mp3','.flac','.ogg'))]
    print(f"\nAvailable voices in {VOICES_DIR}:")
    for f in sorted(files):
        name = os.path.splitext(f)[0]
        print(f"  {name}  ({f})")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VibeVoice Batch Generator")
    parser.add_argument("--input_dir",  required=False, help="Folder with .txt files")
    parser.add_argument("--output_dir", required=False, help="Folder to save .wav files")
    parser.add_argument("--speaker",    default="Alice", help="Speaker voice name")
    parser.add_argument("--cfg_scale",  type=float, default=1.32, help="CFG scale (1.30-1.35)")
    parser.add_argument("--model_path", default="microsoft/VibeVoice-1.5B")
    parser.add_argument("--list_voices", action="store_true", help="List available voices and exit")
    parser.add_argument("--reset",      action="store_true", help="Reset progress (re-generate all files)")
    args = parser.parse_args()

    if args.list_voices:
        list_voices()
        return

    if not args.input_dir or not args.output_dir:
        parser.print_help()
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all txt files
    txt_files = sorted(f for f in os.listdir(args.input_dir) if f.lower().endswith(".txt"))
    if not txt_files:
        print("No .txt files found in input directory.")
        return

    # Progress tracking
    if args.reset and os.path.isfile(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("Progress reset.")

    done_names = load_progress()
    pending    = [f for f in txt_files if os.path.splitext(f)[0] not in done_names]
    total      = len(txt_files)
    already    = total - len(pending)

    print(f"\n{'='*50}")
    print(f"  VibeVoice Batch Generator")
    print(f"  Model:   {args.model_path}")
    print(f"  Speaker: {args.speaker}")
    print(f"  CFG:     {args.cfg_scale}")
    print(f"  Files:   {total} total · {already} done · {len(pending)} remaining")
    print(f"{'='*50}\n")

    if not pending:
        print("All files already converted! Use --reset to regenerate.")
        return

    # Resolve voice path
    voice_path = get_voice_path(args.speaker)
    print(f"Voice file: {voice_path}\n")

    # Load model ONCE
    print("Loading model (this may take 30s)...")
    from transformers import set_seed
    processor = VibeVoiceProcessor.from_pretrained(args.model_path)
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa"
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=10)
    print("Model ready.\n")

    # Load voice audio ONCE
    voice_audio = load_audio(voice_path)

    # Process files
    for idx, fname in enumerate(pending, 1):
        base  = os.path.splitext(fname)[0]
        fpath = os.path.join(args.input_dir, fname)

        print(f"[{idx}/{len(pending)}] {fname}")

        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                raw = fh.read()
        except Exception as e:
            print(f"  ✗ Read error: {e}")
            continue

        script_text = extract_script(raw)
        if not script_text:
            print("  ✗ No script content — skipping.")
            continue

        script = format_script(script_text, "Speaker 1")

        try:
            t0     = time.time()
            inputs = processor(
                text=[script],
                voice_samples=[[voice_audio]],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.cuda()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=args.cfg_scale,
                    tokenizer=processor.tokenizer,
                    generation_config={"do_sample": False},
                    verbose=False,
                )

            elapsed = time.time() - t0

            # Save audio
            out_path = os.path.join(args.output_dir, base + ".wav")
            processor.save_audio(outputs.speech_outputs[0], output_path=out_path)

            # Stats
            sr = 24000
            n_samples = outputs.speech_outputs[0].shape[-1]
            duration  = n_samples / sr
            rtf       = elapsed / duration if duration > 0 else 0
            size_kb   = os.path.getsize(out_path) // 1024

            print(f"  ✓ Saved {base}.wav  |  {duration:.1f}s audio  |  {elapsed:.1f}s gen  |  RTF {rtf:.2f}x  |  {size_kb}KB")

            # Mark done and persist progress
            done_names.add(base)
            save_progress(done_names)

        except torch.cuda.OutOfMemoryError:
            print("  ✗ CUDA out of memory — clearing cache and skipping.")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"  ✗ Generation error: {e}")
            continue

    print(f"\n{'='*50}")
    print(f"  Batch complete!  {len(done_names)}/{total} files converted.")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()