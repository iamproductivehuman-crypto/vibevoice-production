"""
batch_generate.py
─────────────────
Batch-convert a folder of .txt files to audio using VibeVoice.

Model and processor are loaded ONCE at startup and reused for every file.
This means 100+ files are processed without repeated GPU initialisation.

Text file format:
    *title
    Title of the video
    *script
    Script content here...

    (Only text after *script is sent to the TTS engine.)

Usage:
    python batch_generate.py \\
        --input_dir  /path/to/txts \\
        --output_dir /path/to/output \\
        --speaker    Alice \\
        --cfg_scale  1.32 \\
        --model_path microsoft/VibeVoice-1.5B

    python batch_generate.py --list_voices
    python batch_generate.py --input_dir ... --output_dir ... --reset
"""

import argparse
import os
import re
import sys
import time
import json

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
VIBE_DIR      = os.path.join(SCRIPT_DIR, "VibeVoice")
VOICES_DIR    = os.path.join(VIBE_DIR, "demo", "voices")
PROGRESS_FILE = os.path.join(SCRIPT_DIR, ".batch_progress.json")

if VIBE_DIR not in sys.path:
    sys.path.insert(0, VIBE_DIR)

# ── Early imports (no GPU allocation yet) ─────────────────────────────────────
try:
    import torch
    import soundfile as sf
    import numpy as np
    import librosa
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Run 'bash install.sh' to set up the environment.")
    sys.exit(1)

try:
    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
except ImportError as e:
    print(f"ERROR: VibeVoice not importable: {e}")
    print("Run 'bash install.sh' first.")
    sys.exit(1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_script(text: str) -> str:
    """Return only text after the *script marker."""
    match = re.search(r'\*script\s*\n(.*)', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def format_script(text: str, speaker: str = "Speaker 1") -> str:
    """Format plain text into the VibeVoice 'Speaker: line' format."""
    lines = [p.strip() for p in re.split(r'\n\n+|\n', text) if p.strip()]
    if not lines:
        lines = [text.strip()]
    return "\n".join(f"{speaker}: {line}" for line in lines)


def get_voice_path(speaker_name: str) -> str:
    """Resolve a speaker name to a voice file path."""
    if not os.path.isdir(VOICES_DIR):
        raise FileNotFoundError(f"Voices directory not found: {VOICES_DIR}")

    files = [
        f for f in os.listdir(VOICES_DIR)
        if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))
    ]
    if not files:
        raise FileNotFoundError(f"No voice files in {VOICES_DIR}")

    # Build name → path mapping with common separators stripped
    presets: dict[str, str] = {}
    for f in files:
        name = os.path.splitext(f)[0]
        presets[name] = os.path.join(VOICES_DIR, f)
        if "_" in name:
            presets[name.split("_")[0]] = os.path.join(VOICES_DIR, f)
        if "-" in name:
            presets[name.split("-")[-1]] = os.path.join(VOICES_DIR, f)

    # Exact match
    if speaker_name in presets:
        return presets[speaker_name]

    # Case-insensitive partial match
    sl = speaker_name.lower()
    for k, v in presets.items():
        if sl in k.lower() or k.lower() in sl:
            return v

    # Default to first available voice with a warning
    first = os.path.join(VOICES_DIR, files[0])
    print(f"  ⚠  Speaker '{speaker_name}' not found — using {files[0]}")
    return first


def load_audio(path: str, sr: int = 24000) -> np.ndarray:
    wav, orig_sr = sf.read(path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if orig_sr != sr:
        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=sr)
    return wav.astype(np.float32)


def load_progress() -> set:
    if os.path.isfile(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return set(json.load(f))
    return set()


def save_progress(done: set) -> None:
    with open(PROGRESS_FILE, "w") as f:
        json.dump(sorted(done), f)


def list_voices() -> None:
    if not os.path.isdir(VOICES_DIR):
        print(f"Voices directory not found: {VOICES_DIR}")
        return
    files = sorted(
        f for f in os.listdir(VOICES_DIR)
        if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))
    )
    if not files:
        print("No voice files found.")
        return
    print(f"\nAvailable voices in {VOICES_DIR}:")
    for f in files:
        name = os.path.splitext(f)[0]
        print(f"  {name:<24}  ({f})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VibeVoice Batch Generator — loads model once, processes all files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_dir",   required=False, help="Folder containing .txt files")
    parser.add_argument("--output_dir",  required=False, help="Folder to save .wav files")
    parser.add_argument("--speaker",     default="Alice",  help="Speaker voice name (default: Alice)")
    parser.add_argument("--cfg_scale",   type=float, default=1.32, help="CFG scale, 1.30–1.35 (default: 1.32)")
    parser.add_argument("--model_path",  default="microsoft/VibeVoice-1.5B", help="HuggingFace model ID or local path")
    parser.add_argument("--ddpm_steps",  type=int,   default=10, help="DDPM inference steps (default: 10)")
    parser.add_argument("--list_voices", action="store_true", help="Print available voices and exit")
    parser.add_argument("--reset",       action="store_true", help="Ignore previous progress and regenerate all files")
    args = parser.parse_args()

    # ── --list_voices shortcut ────────────────────────────────────────────────
    if args.list_voices:
        list_voices()
        return

    if not args.input_dir or not args.output_dir:
        parser.print_help()
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Discover .txt files ───────────────────────────────────────────────────
    txt_files = sorted(
        f for f in os.listdir(args.input_dir) if f.lower().endswith(".txt")
    )
    if not txt_files:
        print("No .txt files found in input directory.")
        return

    # ── Progress tracking ─────────────────────────────────────────────────────
    if args.reset and os.path.isfile(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("Progress reset — all files will be regenerated.")

    done_names = load_progress()
    pending    = [f for f in txt_files if os.path.splitext(f)[0] not in done_names]
    total      = len(txt_files)
    already    = total - len(pending)

    print(f"\n{'='*54}")
    print(f"  VibeVoice Batch Generator")
    print(f"  Model:      {args.model_path}")
    print(f"  Speaker:    {args.speaker}")
    print(f"  CFG scale:  {args.cfg_scale}")
    print(f"  DDPM steps: {args.ddpm_steps}")
    print(f"  Files:      {total} total  ·  {already} already done  ·  {len(pending)} pending")
    print(f"{'='*54}\n")

    if not pending:
        print("All files already converted. Use --reset to regenerate.")
        return

    # ── Resolve voice ─────────────────────────────────────────────────────────
    try:
        voice_path = get_voice_path(args.speaker)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    print(f"Voice file:  {voice_path}")

    # ── Load model ONCE ───────────────────────────────────────────────────────
    print(f"\nLoading processor from {args.model_path}...")
    t0 = time.time()
    processor = VibeVoiceProcessor.from_pretrained(args.model_path)
    print(f"  Processor ready  ({time.time()-t0:.1f}s)")

    print(f"Loading model...")
    t0 = time.time()
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=args.ddpm_steps)
    print(f"  Model ready  ({time.time()-t0:.1f}s)\n")

    # ── Load voice audio ONCE ─────────────────────────────────────────────────
    voice_audio = load_audio(voice_path)

    # ── Batch loop ────────────────────────────────────────────────────────────
    batch_start = time.time()
    success     = 0
    skipped_err = 0

    for idx, fname in enumerate(pending, 1):
        base  = os.path.splitext(fname)[0]
        fpath = os.path.join(args.input_dir, fname)
        prefix = f"[{idx}/{len(pending)}]"

        print(f"{prefix} {fname}")

        # Read file
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                raw = fh.read()
        except Exception as e:
            print(f"  ✗ Read error: {e}")
            skipped_err += 1
            continue

        script_text = extract_script(raw)
        if not script_text:
            print("  ✗ No script content found — skipping.")
            skipped_err += 1
            continue

        script = format_script(script_text, "Speaker 1")

        # Generate
        try:
            t_gen  = time.time()
            inputs = processor(
                text=[script],
                voice_samples=[[voice_audio]],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            # Move tensors to GPU
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

            elapsed = time.time() - t_gen

            # Save
            out_path = os.path.join(args.output_dir, base + ".wav")
            processor.save_audio(outputs.speech_outputs[0], output_path=out_path)

            # Stats
            sr        = 24000
            n_samples = outputs.speech_outputs[0].shape[-1]
            duration  = n_samples / sr
            rtf       = elapsed / duration if duration > 0 else 0
            size_kb   = os.path.getsize(out_path) // 1024

            print(
                f"  ✓ {base}.wav  "
                f"dur={duration:.1f}s  gen={elapsed:.1f}s  "
                f"RTF={rtf:.2f}x  {size_kb}KB"
            )

            done_names.add(base)
            save_progress(done_names)
            success += 1

        except torch.cuda.OutOfMemoryError:
            print("  ✗ CUDA OOM — clearing cache and skipping this file.")
            print("  ⚠  Consider using a GPU with more VRAM (RTX 3090/4090 recommended).")
            torch.cuda.empty_cache()
            skipped_err += 1
            continue
        except Exception as e:
            print(f"  ✗ Generation error: {e}")
            skipped_err += 1
            continue

    # ── Teardown (after ALL files) ────────────────────────────────────────────
    del model
    torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.time() - batch_start
    print(f"\n{'='*54}")
    print(f"  Batch complete!")
    print(f"  {success} succeeded  ·  {skipped_err} skipped/errored  ·  {already} already done")
    print(f"  Total time: {total_elapsed/60:.1f} min")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*54}")


if __name__ == "__main__":
    main()