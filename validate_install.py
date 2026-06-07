"""
validate_install.py
───────────────────
Runs a real end-to-end test: loads the model, generates a short audio clip,
and saves it as test_output.wav.

Called automatically by install.sh (step 8).
Can also be run manually at any time:
    python validate_install.py

Exit codes:
    0 — test generation succeeded
    1 — something failed; read the error above
"""

import os
import sys
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIBE_DIR   = os.path.join(SCRIPT_DIR, "VibeVoice")
VOICES_DIR = os.path.join(VIBE_DIR, "demo", "voices")
OUTPUT_WAV = os.path.join(SCRIPT_DIR, "test_output.wav")
MODEL_ID   = "microsoft/VibeVoice-1.5B"

# ── Ensure VibeVoice is on sys.path ──────────────────────────────────────────
if VIBE_DIR not in sys.path:
    sys.path.insert(0, VIBE_DIR)

print("=" * 54)
print("  VibeVoice Install Validation")
print("=" * 54)

# ── 1. Imports ────────────────────────────────────────────────────────────────
print("\n[1/5] Importing packages...")
try:
    import torch
    print(f"  torch {torch.__version__}")
except ImportError as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

try:
    import soundfile as sf
    import librosa
    print("  soundfile + librosa  OK")
except ImportError as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

try:
    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    print("  VibeVoice classes imported  OK")
except ImportError as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# ── 2. GPU ────────────────────────────────────────────────────────────────────
print("\n[2/5] Checking GPU...")
if not torch.cuda.is_available():
    print("  FAIL: No CUDA GPU available.")
    sys.exit(1)

gpu   = torch.cuda.get_device_name(0)
vram  = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  {gpu}  {vram:.1f} GB  OK")

# ── 3. Pick a voice sample ────────────────────────────────────────────────────
print("\n[3/5] Locating voice sample...")

voice_path = None
if os.path.isdir(VOICES_DIR):
    for f in sorted(os.listdir(VOICES_DIR)):
        if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
            voice_path = os.path.join(VOICES_DIR, f)
            print(f"  Using voice: {f}")
            break

if voice_path is None:
    # No real voice — synthesise a short silent placeholder so the test runs
    print("  ⚠  No voice files found in demo/voices/.")
    print("  ⚠  Creating a 3-second silent placeholder for the test.")
    SR = 24000
    silence = np.zeros(SR * 3, dtype=np.float32)
    placeholder = os.path.join(SCRIPT_DIR, "_test_placeholder.wav")
    sf.write(placeholder, silence, SR)
    voice_path = placeholder

# Load voice audio
def load_audio(path, sr=24000):
    wav, orig_sr = sf.read(path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if orig_sr != sr:
        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=sr)
    return wav.astype(np.float32)

voice_audio = load_audio(voice_path)

# ── 4. Load processor + model ────────────────────────────────────────────────
print("\n[4/5] Loading processor and model...")
t_load = time.time()

try:
    processor = VibeVoiceProcessor.from_pretrained(MODEL_ID)
    print("  Processor loaded  OK")
except Exception as e:
    print(f"  FAIL loading processor: {e}")
    sys.exit(1)

try:
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=10)
    load_elapsed = time.time() - t_load
    print(f"  Model loaded  OK  ({load_elapsed:.1f}s)")
except Exception as e:
    print(f"  FAIL loading model: {e}")
    sys.exit(1)

# ── 5. Generate ───────────────────────────────────────────────────────────────
print("\n[5/5] Generating test audio...")
TEST_TEXT = "Speaker 1: This is a test. VibeVoice is working correctly."

try:
    t_gen = time.time()
    inputs = processor(
        text=[TEST_TEXT],
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
            cfg_scale=1.32,
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
        )

    gen_elapsed = time.time() - t_gen
    processor.save_audio(outputs.speech_outputs[0], output_path=OUTPUT_WAV)

    n_samples = outputs.speech_outputs[0].shape[-1]
    duration  = n_samples / 24000
    size_kb   = os.path.getsize(OUTPUT_WAV) // 1024
    rtf       = gen_elapsed / duration if duration > 0 else 0

    print(f"  Audio saved: {OUTPUT_WAV}")
    print(f"  Duration: {duration:.1f}s  |  Generated in: {gen_elapsed:.1f}s  |  RTF: {rtf:.2f}x  |  Size: {size_kb} KB")

except torch.cuda.OutOfMemoryError:
    print("  FAIL: CUDA out of memory during test generation.")
    print("  Your GPU may not have enough VRAM. Recommended: RTX 3090 / RTX 4090 (24 GB).")
    del model
    torch.cuda.empty_cache()
    sys.exit(1)
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# ── Cleanup ───────────────────────────────────────────────────────────────────
del model
torch.cuda.empty_cache()

# Remove silent placeholder if we created one
placeholder = os.path.join(SCRIPT_DIR, "_test_placeholder.wav")
if os.path.isfile(placeholder):
    os.remove(placeholder)

print()
print("=" * 54)
print("  ✓ Installation verified successfully.")
print(f"  ✓ Test audio: {OUTPUT_WAV}")
print("=" * 54)
