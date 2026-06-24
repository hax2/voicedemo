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

# 3. Install accent trainer dependencies
echo "[3/4] Installing accent trainer dependencies..."
pip install -r requirements.txt
pip install --upgrade "protobuf>=3.20.3"

# 4. Create cache directories
echo "[4/4] Creating cache directories..."
mkdir -p cached_tts
mkdir -p cached_vc
mkdir -p checkpoints

echo ""
echo "=== Setup Complete ==="
echo "Run the app with: python app.py"
echo "The Gradio share link will be printed to the console."
