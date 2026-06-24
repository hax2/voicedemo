#!/bin/bash
set -e

echo "=== Accent Trainer - Server Setup ==="
echo ""

# 1. Clone seed-vc if not already present
if [ ! -d "seed-vc" ]; then
    echo "[1/4] Cloning seed-vc repository..."
    git clone https://github.com/Plachtaa/seed-vc.git
else
    echo "[1/4] seed-vc already cloned, skipping."
fi

# 2. Install seed-vc dependencies
echo "[2/4] Installing seed-vc dependencies..."
cd seed-vc
pip install -r requirements.txt
cd ..

# 3. Clone EZ-VC if not already present
if [ ! -d "EZ-VC" ]; then
    echo "[3/6] Cloning EZ-VC repository..."
    git clone https://github.com/EZ-VC/EZ-VC.git
    cd EZ-VC
    git submodule update --init --recursive
    cd ..
else
    echo "[3/6] EZ-VC already cloned, skipping."
fi

# 4. Install EZ-VC dependencies
echo "[4/6] Installing EZ-VC dependencies..."
cd EZ-VC
git config --global --add safe.directory "$PWD"
pip install -e .
pip install 'espnet @ git+https://github.com/wanchichen/espnet.git@ssl'
cd ..

# 5. Install accent trainer dependencies
echo "[5/6] Installing accent trainer dependencies..."
pip install -r requirements.txt
pip install cached-path
pip install --upgrade protobuf

# 6. Create cache directories
echo "[6/6] Creating cache directories..."
mkdir -p cached_tts
mkdir -p cached_vc
mkdir -p checkpoints

echo ""
echo "=== Setup Complete ==="
echo "Run the app with: python app.py"
echo "The Gradio share link will be printed to the console."
