"""
verify.py
─────────
Validates the VibeVoice installation and prints a full environment report.
Run after install.sh, or any time you want to confirm the environment is healthy.

Exit codes:
    0  — all checks passed
    1  — one or more checks failed
"""

import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIBE_DIR   = os.path.join(SCRIPT_DIR, "VibeVoice")
VOICES_DIR = os.path.join(VIBE_DIR, "demo", "voices")

REQUIRED_TRANSFORMERS = "4.51.3"

CHECK  = "✓"
CROSS  = "✗"
WARN   = "⚠"

results = []

def ok(label, detail=""):
    msg = f"  {CHECK} {label}"
    if detail:
        msg += f"  [{detail}]"
    print(msg)
    results.append(True)

def fail(label, detail=""):
    msg = f"  {CROSS} {label}"
    if detail:
        msg += f"  [{detail}]"
    print(msg)
    results.append(False)

def warn(label, detail=""):
    msg = f"  {WARN} {label}"
    if detail:
        msg += f"  [{detail}]"
    print(msg)
    # warnings don't count as failures

print("=" * 54)
print("  VibeVoice Environment Verification")
print("=" * 54)

# ── GPU ───────────────────────────────────────────────
print("\n[GPU]")
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        mem_gb   = torch.cuda.get_device_properties(0).total_memory / 1e9
        cuda_ver = torch.version.cuda
        ok("GPU detected", f"{gpu_name}  {mem_gb:.1f} GB  CUDA {cuda_ver}")
    else:
        fail("No CUDA GPU detected — model requires GPU")
except Exception as e:
    fail("torch.cuda check failed", str(e))

# ── CUDA driver ───────────────────────────────────────
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        stderr=subprocess.DEVNULL,
    ).decode().strip()
    ok("NVIDIA driver", out)
except Exception:
    warn("nvidia-smi not available")

# ── Python / Torch ────────────────────────────────────
print("\n[Versions]")
ok("Python", sys.version.split()[0])

try:
    import torch
    ok("torch", torch.__version__)
except ImportError:
    fail("torch not installed")

# ── Transformers version ──────────────────────────────
try:
    import transformers
    ver = transformers.__version__
    if ver == REQUIRED_TRANSFORMERS:
        ok("transformers", ver)
    else:
        fail(
            f"transformers version mismatch  (need {REQUIRED_TRANSFORMERS}, got {ver})",
            "Re-run: pip install transformers==" + REQUIRED_TRANSFORMERS,
        )
except ImportError:
    fail("transformers not installed")

# ── Other packages ────────────────────────────────────
for pkg in ("accelerate", "soundfile", "librosa", "numpy", "diffusers"):
    try:
        mod = __import__(pkg)
        ok(pkg, getattr(mod, "__version__", "?"))
    except ImportError:
        fail(f"{pkg} not installed")

# ── VibeVoice repo ────────────────────────────────────
print("\n[VibeVoice Repository]")
if os.path.isdir(VIBE_DIR):
    ok("VibeVoice directory found", VIBE_DIR)
    # Get commit hash
    try:
        commit = subprocess.check_output(
            ["git", "-C", VIBE_DIR, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        ok("VibeVoice commit", commit)
    except Exception:
        warn("Could not read git commit")
else:
    fail("VibeVoice directory missing — run install.sh")

# ── Critical file ─────────────────────────────────────
inference_file = os.path.join(
    VIBE_DIR,
    "vibevoice", "modular", "modeling_vibevoice_inference.py"
)
if os.path.isfile(inference_file):
    ok("modeling_vibevoice_inference.py present")
else:
    fail(
        "modeling_vibevoice_inference.py NOT found",
        "Wrong repo version — check VIBE_COMMIT in install.sh",
    )

# ── Package importability ─────────────────────────────
print("\n[VibeVoice Import]")
if VIBE_DIR not in sys.path:
    sys.path.insert(0, VIBE_DIR)

try:
    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,
    )
    ok("VibeVoiceForConditionalGenerationInference imported")
except Exception as e:
    fail("Model import failed", str(e))

try:
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    ok("VibeVoiceProcessor imported")
except Exception as e:
    fail("Processor import failed", str(e))

# ── Voices ────────────────────────────────────────────
print("\n[Voices]")
if os.path.isdir(VOICES_DIR):
    voices = [
        f for f in os.listdir(VOICES_DIR)
        if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))
    ]
    if voices:
        ok(f"{len(voices)} voice file(s) found", VOICES_DIR)
        for v in sorted(voices):
            print(f"       - {v}")
    else:
        warn("No voice files found in demo/voices/ — add .wav/.mp3 files")
else:
    warn(f"Voices directory not found: {VOICES_DIR}")

# ── Summary ───────────────────────────────────────────
print()
print("=" * 54)
passed = sum(results)
total  = len(results)
if all(results):
    print(f"  {CHECK} All {total} checks passed — ready for generation!")
else:
    failed = total - passed
    print(f"  {CROSS} {failed}/{total} check(s) FAILED — fix errors above before running batch_generate.py")
print("=" * 54)

sys.exit(0 if all(results) else 1)
