"""
verify.py
─────────
Prints a full environment report and writes setup_report.txt.
Run any time to confirm the environment is healthy.

    python verify.py

Exit codes:
    0 — all checks passed
    1 — one or more checks failed
"""

import os
import sys
import subprocess
import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIBE_DIR   = os.path.join(SCRIPT_DIR, "VibeVoice")
VOICES_DIR = os.path.join(VIBE_DIR, "demo", "voices")
REPORT_FILE = os.path.join(SCRIPT_DIR, "setup_report.txt")

REQUIRED_TORCH        = "2.5.1"
REQUIRED_TRANSFORMERS = "4.51.3"
# accelerate: VibeVoice declares no version requirement; transformers 4.51.3
# requires only >=0.26.0.  We have verified that 1.6.0 works end-to-end
# (imports, model load, generation all pass).  Accept any 1.x release ≥ 1.6.0.
ACCELERATE_MIN        = (1, 6, 0)
VRAM_WARN_GB          = 16

CHECK = "✓"
CROSS = "✗"
WARN  = "⚠"

results      = []   # True = pass, False = fail
report_lines = []

def p(line=""):
    """Print and also buffer for the report file."""
    print(line)
    report_lines.append(line)

def ok(label, detail=""):
    msg = f"  {CHECK} {label}"
    if detail:
        msg += f"  [{detail}]"
    p(msg)
    results.append(True)

def fail(label, detail=""):
    msg = f"  {CROSS} {label}"
    if detail:
        msg += f"  [{detail}]"
    p(msg)
    results.append(False)

def warn(label, detail=""):
    msg = f"  {WARN} {label}"
    if detail:
        msg += f"  [{detail}]"
    p(msg)
    # warnings are informational — they don't fail the check

def ver_check(pkg_name, installed, required):
    """Exact version match check."""
    if installed == required:
        ok(pkg_name, installed)
    else:
        fail(
            f"{pkg_name} version mismatch",
            f"need {required}, got {installed}  →  pip install {pkg_name}=={required}",
        )

# ─────────────────────────────────────────────────────────────────────────────
p("=" * 56)
p("  VibeVoice Environment Report")
p(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
p("=" * 56)

# ── GPU ───────────────────────────────────────────────────────────────────────
p("\n[GPU]")
try:
    import torch
    if torch.cuda.is_available():
        props    = torch.cuda.get_device_properties(0)
        gpu_name = torch.cuda.get_device_name(0)
        mem_gb   = props.total_memory / 1e9
        cuda_ver = torch.version.cuda
        ok("GPU detected", f"{gpu_name}  {mem_gb:.1f} GB  CUDA {cuda_ver}")
        if mem_gb < VRAM_WARN_GB:
            warn(
                f"VRAM only {mem_gb:.0f} GB",
                f"Recommended ≥{VRAM_WARN_GB} GB (RTX 3090/4090) to avoid OOM",
            )
    else:
        fail("No CUDA GPU detected", "model requires a CUDA GPU")
except Exception as e:
    fail("torch.cuda check failed", str(e))

try:
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        stderr=subprocess.DEVNULL,
    ).decode().strip()
    ok("NVIDIA driver", driver)
except Exception:
    warn("nvidia-smi not available")

# ── Python ────────────────────────────────────────────────────────────────────
p("\n[Versions]")
ok("Python", sys.version.split()[0])

# ── Torch ─────────────────────────────────────────────────────────────────────
try:
    import torch
    # Accept patch variants, e.g. 2.5.1+cu121
    base = torch.__version__.split("+")[0]
    ver_check("torch", base, REQUIRED_TORCH)
except ImportError:
    fail("torch not installed")

# ── Transformers (exact) ──────────────────────────────────────────────────────
try:
    import transformers
    ver_check("transformers", transformers.__version__, REQUIRED_TRANSFORMERS)
except ImportError:
    fail("transformers not installed")

# ── Accelerate (minimum floor, not exact pin) ─────────────────────────────────
# Neither VibeVoice nor transformers 4.51.3 require a specific accelerate patch
# release.  We accept any version >= ACCELERATE_MIN that passes real functional
# tests (imports, model load, generation).
try:
    import accelerate
    from packaging.version import Version
    acc_ver = accelerate.__version__
    acc_tuple = tuple(int(x) for x in acc_ver.split(".")[:3])
    min_str = ".".join(str(x) for x in ACCELERATE_MIN)
    if acc_tuple >= ACCELERATE_MIN:
        ok("accelerate", f"{acc_ver}  (>= {min_str} required)")
    else:
        fail(
            f"accelerate too old  (need >= {min_str}, got {acc_ver})",
            f"pip install 'accelerate>={min_str}'",
        )
except ImportError:
    fail("accelerate not installed")

# ── Other required packages ───────────────────────────────────────────────────
for pkg in ("soundfile", "librosa", "numpy", "diffusers", "paramiko"):
    try:
        mod = __import__(pkg)
        ok(pkg, getattr(mod, "__version__", "present"))
    except ImportError:
        fail(f"{pkg} not installed")

# ── VibeVoice repository ──────────────────────────────────────────────────────
p("\n[VibeVoice Repository]")
if os.path.isdir(VIBE_DIR):
    ok("VibeVoice directory", VIBE_DIR)
    try:
        commit = subprocess.check_output(
            ["git", "-C", VIBE_DIR, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        ok("VibeVoice commit", commit)
    except Exception:
        warn("Could not read git commit")
else:
    fail("VibeVoice directory not found", "run install.sh")

# ── Critical source file ──────────────────────────────────────────────────────
inference_file = os.path.join(
    VIBE_DIR, "vibevoice", "modular", "modeling_vibevoice_inference.py"
)
if os.path.isfile(inference_file):
    ok("modeling_vibevoice_inference.py present")
else:
    fail(
        "modeling_vibevoice_inference.py NOT found",
        "Wrong repo version or commit — update VIBE_COMMIT in install.sh",
    )

# ── Package import check ──────────────────────────────────────────────────────
p("\n[VibeVoice Import]")
if VIBE_DIR not in sys.path:
    sys.path.insert(0, VIBE_DIR)

try:
    import vibevoice
    ok("vibevoice package importable", vibevoice.__file__)
except Exception as e:
    fail("vibevoice import failed", str(e))

try:
    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,
    )
    ok("VibeVoiceForConditionalGenerationInference imported")
except Exception as e:
    fail("Model class import failed", str(e))

try:
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    ok("VibeVoiceProcessor imported")
except Exception as e:
    fail("Processor import failed", str(e))

# ── Voices ────────────────────────────────────────────────────────────────────
p("\n[Voices]")
if os.path.isdir(VOICES_DIR):
    voices = sorted(
        f for f in os.listdir(VOICES_DIR)
        if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))
    )
    if voices:
        ok(f"{len(voices)} voice file(s) found", VOICES_DIR)
        for v in voices:
            p(f"       - {v}")
    else:
        warn("No voice files in demo/voices/ — add .wav/.mp3 before generating")
else:
    warn(f"Voices directory not found: {VOICES_DIR}")

# ── Test output ───────────────────────────────────────────────────────────────
p("\n[Test Generation]")
test_wav = os.path.join(SCRIPT_DIR, "test_output.wav")
if os.path.isfile(test_wav):
    size_kb = os.path.getsize(test_wav) // 1024
    ok("test_output.wav present", f"{size_kb} KB")
else:
    warn("test_output.wav not found", "run validate_install.py to generate it")

# ── Summary ───────────────────────────────────────────────────────────────────
p("")
p("=" * 56)
passed = sum(results)
total  = len(results)
failed = total - passed

if all(results):
    p(f"  {CHECK} All {total} checks passed — ready for generation!")
else:
    p(f"  {CROSS} {failed}/{total} check(s) FAILED — see details above.")

p("=" * 56)

# ── Write setup_report.txt ────────────────────────────────────────────────────
try:
    with open(REPORT_FILE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines) + "\n")
    print(f"\n  Report written to: {REPORT_FILE}")
except Exception as e:
    print(f"\n  (Could not write report file: {e})")

sys.exit(0 if all(results) else 1)