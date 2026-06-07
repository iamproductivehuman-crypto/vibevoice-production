"""
patch.py — Auto-fixes known issues in VibeVoice source files.

Safe to run on any clone of the repo: every patch is guarded with
an existence check, so missing files are skipped cleanly.

Runs from the PARENT of the VibeVoice directory (default) or from
inside VibeVoice itself.

    python patch.py            # from vibevoice-production/
    python ../patch.py         # from vibevoice-production/VibeVoice/
"""
import os
import sys

# ── Locate VibeVoice root ─────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Called from the repo root → VibeVoice is a subdirectory
VIBE_ROOT = os.path.join(SCRIPT_DIR, "VibeVoice")

# Called from inside VibeVoice → SCRIPT_DIR is already the root
if not os.path.isdir(VIBE_ROOT):
    # Check if there's a vibevoice package here instead
    if os.path.isdir(os.path.join(SCRIPT_DIR, "vibevoice")):
        VIBE_ROOT = SCRIPT_DIR

if not os.path.isdir(VIBE_ROOT):
    print(f"ERROR: Cannot locate VibeVoice directory.")
    print(f"  Looked at: {VIBE_ROOT}")
    sys.exit(1)

print("=" * 54)
print("  VibeVoice — Applying Source Patches")
print(f"  Root: {VIBE_ROOT}")
print("=" * 54)


def patch_file(rel_path, replacements, description=""):
    """Apply (old, new) replacements to a file.

    - Skips gracefully if the file does not exist.
    - Reports every replacement attempt (found / already patched).
    - Returns True if the file was written.
    """
    full_path = os.path.join(VIBE_ROOT, rel_path)
    if not os.path.isfile(full_path):
        print(f"\n  [SKIP] {rel_path}  (not in this repo version)")
        return False

    print(f"\n  [PATCH] {rel_path}" + (f" — {description}" if description else ""))

    with open(full_path, "r", encoding="utf-8") as fh:
        content = fh.read()

    original = content
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            show_old = repr(old[:72])
            show_new = repr(new[:52]) if new else "''"
            print(f"    ✓  {show_old}")
            print(f"       → {show_new}")
        else:
            print(f"    ~  Not found (already patched?): {repr(old[:72])}")

    if content != original:
        with open(full_path, "w", encoding="utf-8") as fh:
            fh.write(content)
        print(f"    Saved.")
        return True
    else:
        print(f"    No changes needed.")
        return False


# ── Shared patch sets ─────────────────────────────────────────────────────────

# flash_attention_2 → sdpa  (sdpa ships with every PyTorch ≥ 2.0)
FLASH_PATCHES = [
    ('attn_implementation="flash_attention_2"', 'attn_implementation="sdpa"'),
    ("attn_implementation='flash_attention_2'", "attn_implementation='sdpa'"),
    # also catch bare string assignments, just in case
    ('"flash_attention_2"',                     '"sdpa"'),
]

GRADIO_PATCHES = FLASH_PATCHES + [
    # Gradio ≥ 4.x dropped these kwargs
    ("show_copy_button=True,",   ""),
    ("show_copy_button=True",    ""),
    ("theme=gr.themes.Soft(),",  ""),
    ("theme=gr.themes.Soft()",   ""),
    ("css=custom_css,",          ""),
    ("css=custom_css",           ""),
]

# ── Patch list ────────────────────────────────────────────────────────────────
# (relative_path_from_VibeVoice_root, patch_set, description)
PATCHES = [
    # ── demo scripts ─────────────────────────────────────────────────────────
    ("demo/inference_from_file.py",
     FLASH_PATCHES,
     "flash_attention_2 → sdpa"),

    ("demo/inference_from_file_streaming.py",
     FLASH_PATCHES,
     "flash_attention_2 → sdpa"),

    ("demo/gradio_demo.py",
     GRADIO_PATCHES,
     "flash_attention_2 → sdpa, Gradio API fixes"),

    ("demo/colab.py",
     FLASH_PATCHES,
     "flash_attention_2 → sdpa"),

    # ── modular inference (present in harry2141985 fork) ──────────────────────
    ("vibevoice/modular/modeling_vibevoice_inference.py",
     FLASH_PATCHES,
     "flash_attention_2 → sdpa"),

    ("vibevoice/modular/modeling_vibevoice_streaming_inference.py",
     FLASH_PATCHES,
     "flash_attention_2 → sdpa"),

    # ── any top-level inference script ───────────────────────────────────────
    ("inference.py",
     FLASH_PATCHES,
     "flash_attention_2 → sdpa"),
]

# ── Run patches ───────────────────────────────────────────────────────────────
modified = 0
skipped  = 0

for rel, patch_set, desc in PATCHES:
    changed = patch_file(rel, patch_set, desc)
    if changed:
        modified += 1
    elif not os.path.isfile(os.path.join(VIBE_ROOT, rel)):
        skipped += 1

print()
print("=" * 54)
print(f"  Done: {modified} file(s) modified, {skipped} skipped.")
print("=" * 54)