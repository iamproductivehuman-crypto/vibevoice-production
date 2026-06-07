#!/bin/bash
set -e

echo "=================================================="
echo "  VibeVoice Production Setup"
echo "=================================================="

# ── 1. Clone VibeVoice ────────────────────────────────
if [ ! -d "VibeVoice" ]; then
    echo "[1/6] Cloning VibeVoice..."
    git clone https://github.com/microsoft/VibeVoice.git
else
    echo "[1/6] VibeVoice already cloned, skipping."
fi

cd VibeVoice

# ── 2. Install dependencies ───────────────────────────
echo "[2/6] Installing dependencies..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q -r requirements.txt
pip install -q transformers accelerate soundfile librosa paramiko

# ── 3. Apply patches ──────────────────────────────────
echo "[3/6] Applying patches..."
python ../patch.py

# ── 4. Download model ────────────────────────────────
echo "[4/6] Downloading model (this takes 5-10 min)..."
python -c "
from transformers import AutoConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
import torch
model = VibeVoiceForConditionalGenerationInference.from_pretrained(
    'microsoft/VibeVoice-1.5B',
    torch_dtype=torch.bfloat16,
    device_map='cuda',
    attn_implementation='sdpa'
)
print('Model loaded OK')
del model
import torch; torch.cuda.empty_cache()
"

# ── 5. Copy voices ────────────────────────────────────
echo "[5/6] Copying voices..."
mkdir -p demo/voices
cp ../voices/* demo/voices/ 2>/dev/null || echo "  No custom voices found in ../voices/, using defaults."

# ── 6. Verify ─────────────────────────────────────────
echo "[6/6] Verifying setup..."
python -c "
import os
voices_dir = 'demo/voices'
voices = [f for f in os.listdir(voices_dir) if f.endswith(('.wav','.mp3'))]
print(f'  GPU: ' + __import__('torch').cuda.get_device_name(0))
print(f'  Voices found: {len(voices)}')
for v in voices: print(f'    - {v}')
"

cd ..
echo ""
echo "=================================================="
echo "  Setup complete!"
echo "  Run: python batch_generate.py --input_dir /path/to/txts --output_dir /path/to/output"
echo "=================================================="
