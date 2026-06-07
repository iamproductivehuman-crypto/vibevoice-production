"""
patch.py — Auto-fixes all known issues in VibeVoice demo files.
Run from inside the VibeVoice directory after cloning.
"""
import os
import re

def patch_file(path, replacements, description=""):
    if not os.path.isfile(path):
        print(f"  SKIP (not found): {path}")
        return
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    original = content
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            print(f"  ✓ Patched: {repr(old[:60])} → {repr(new[:40])}")
        else:
            print(f"  ~ Already patched or not found: {repr(old[:60])}")
    if content != original:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Saved: {path}")
    else:
        print(f"  No changes needed: {path}")

print("=" * 50)
print("Applying VibeVoice patches...")
print("=" * 50)

# ── Fix 1: inference_from_file.py ────────────────────
print("\n[1] demo/inference_from_file.py")
patch_file("demo/inference_from_file.py", [
    # Use sdpa instead of flash_attention_2 (not always installed)
    ('attn_implementation="flash_attention_2"', 'attn_implementation="sdpa"'),
    ("attn_implementation='flash_attention_2'", "attn_implementation='sdpa'"),
])

# ── Fix 2: gradio_demo.py ─────────────────────────────
print("\n[2] demo/gradio_demo.py")
patch_file("demo/gradio_demo.py", [
    # flash_attention fix
    ('attn_implementation="flash_attention_2"', 'attn_implementation="sdpa"'),
    ("attn_implementation='flash_attention_2'", "attn_implementation='sdpa'"),
    # Gradio API compatibility
    ('show_copy_button=True', ''),
    ('theme=gr.themes.Soft()', ''),
    ('theme=gr.themes.Soft(),', ''),
    ('css=custom_css,', ''),
    ('css=custom_css', ''),
])

# ── Fix 3: colab.py ────────────────────────────────────
print("\n[3] demo/colab.py")
patch_file("demo/colab.py", [
    ('attn_implementation="flash_attention_2"', 'attn_implementation="sdpa"'),
    ("attn_implementation='flash_attention_2'", "attn_implementation='sdpa'"),
])

print("\n" + "=" * 50)
print("All patches applied.")
print("=" * 50)
