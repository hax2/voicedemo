#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Accent Trainer - Server Setup ==="
echo ""

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x ".venv/bin/python" ]; then
        PYTHON_BIN=".venv/bin/python"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        PYTHON_BIN="python3"
    fi
fi

echo "Using Python interpreter: $PYTHON_BIN"

# 1. Clone Seed-VC if not already present
if [ ! -d "seed-vc" ]; then
    echo "[1/7] Cloning seed-vc repository..."
    git clone https://github.com/Plachtaa/seed-vc.git
else
    echo "[1/7] seed-vc already cloned, skipping."
fi

# 2. Clone EZ-VC if not already present
if [ ! -d "EZ-VC" ]; then
    echo "[2/7] Cloning EZ-VC repository..."
    git clone https://github.com/EZ-VC/EZ-VC.git
    (cd EZ-VC && git submodule update --init --recursive)
else
    echo "[2/7] EZ-VC already cloned, ensuring submodules..."
    (cd EZ-VC && git submodule update --init --recursive)
fi

# 3. Clone OmniVoice if not already present
if [ ! -d "OmniVoice" ]; then
    echo "[3/7] Cloning OmniVoice repository..."
    git clone https://github.com/k2-fsa/OmniVoice.git
else
    echo "[3/7] OmniVoice already cloned, skipping."
fi

# 4. Install unified core & model dependencies without version tug-of-wars
echo "[4/7] Installing unified Python dependencies..."
git config --global --add safe.directory '*' || true

"$PYTHON_BIN" -m pip install --upgrade pip "setuptools<81" wheel packaging hatchling editables setuptools-scm pytest-runner

"$PYTHON_BIN" -m pip install \
    "torch==2.6.0" "torchaudio==2.6.0" "torchvision==0.21.0" \
    "numpy>=1.24.0,<2.0.0" "pyarrow<20.0.0" "datasets<4.0.0" \
    "scipy>=1.11.0" "librosa>=0.10.2" "soundfile>=0.12.1" \
    "matplotlib>=3.7.0" "requests>=2.31.0" "pydub>=0.25.1" \
    "gradio>=5.23.0,<6.0.0" "pydantic<=2.10.6" "protobuf>=4.25.0,<6.0.0" \
    "hydra-core>=1.3.2" omegaconf pyyaml "python-dotenv>=1.0.0" munch "einops>=0.8.0" \
    "accelerate>=0.33.0" "transformers>=5.3.0" "huggingface-hub>=0.34.0" hf_transfer \
    tensorboardX webdataset \
    cached-path click "ema-pytorch>=0.5.2" jieba pypinyin safetensors tomli torchdiffeq \
    "tqdm>=4.65.0" transformers-stream-generator unidecode vocos wandb \
    "x-transformers==1.31.14" torch-einops-utils bitsandbytes \
    configargparse humanfriendly kaldiio torch-complex opt-einsum h5py \
    sentencepiece scikit-learn joblib nltk editdistance jamo \
    espnet-tts-frontend g2p-en

# 5. Install EZ-VC, OmniVoice, and ESPnet SSL without overriding pinned versions
echo "[5/7] Installing EZ-VC, OmniVoice, and ESPnet SSL packages..."
"$PYTHON_BIN" -m pip install --no-deps --no-build-isolation \
    -e ./EZ-VC \
    -e ./OmniVoice \
    'espnet @ git+https://github.com/wanchichen/espnet.git@ssl'

# 6. Create symlink for EZ-VC xeus configs & cache directories
echo "[6/7] Configuring EZ-VC xeus paths and cache directories..."
ln -sfn EZ-VC/src/f5_tts/infer/xeus xeus
mkdir -p cached_tts cached_vc checkpoints

# 7. Verify installation
echo "[7/7] Verifying environment..."
"$PYTHON_BIN" - <<'PY'
import torch
import torchaudio
import gradio
import numpy
import transformers

print(f"Verified torch        {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
print(f"Verified torchaudio   {torchaudio.__version__}")
print(f"Verified transformers {transformers.__version__}")
print(f"Verified gradio       {gradio.__version__}")
print(f"Verified numpy        {numpy.__version__}")
PY

echo ""
echo "=== Setup Complete ==="
echo "Run the app with: python app.py"
echo "Note: For EZ-VC, accept access at https://huggingface.co/SPRINGLab/EZ-VC and set HF_TOKEN in .env."
