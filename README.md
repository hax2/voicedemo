# 🎯 Accent Trainer — Seed-VC v2 Demo

AI-powered accent training for language learning. Record your voice, and the system generates ideal pronunciation examples **in your own voice** using zero-shot voice conversion.

**Languages:** English ↔ Spanish

## How It Works

```
Your Voice Sample ─────────────────────────┐
                                           ▼
Google TTS (Chirp 3 HD) ──► Seed-VC v2 ──► Your Voice Speaking Correctly
      (correct accent)     (voice clone)    (ideal pronunciation)
```

1. **Enroll** — Record 10-25s of your natural speech
2. **Generate** — TTS creates a perfect pronunciation → Seed-VC v2 converts it to your voice
3. **Practice** — Record yourself attempting the sentence
4. **Compare** — View side-by-side spectrograms, waveforms, and pitch contours

## Prerequisites

- Python 3.10
- NVIDIA GPU (≥8GB VRAM) — tested on A100 40GB
- FFmpeg installed system-wide
- Google Cloud TTS API key

## Quick Start (Server)

```bash
# 1. Clone this repo
git clone <your-repo-url>
cd voice

# 2. Run the setup script
chmod +x setup_server.sh
bash setup_server.sh

# 3. Launch the app
python app.py --api-key YOUR_GOOGLE_API_KEY

# The Gradio share link will be printed — access from any device!
```

## Manual Setup

```bash
# Clone seed-vc
git clone https://github.com/Plachtaa/seed-vc.git
cd seed-vc && pip install -r requirements.txt && cd ..

# Install accent trainer deps
pip install -r requirements.txt

# Run
python app.py --api-key YOUR_API_KEY
```

## Command Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--api-key` | env var | Google Cloud TTS API key |
| `--seed-vc-path` | `./seed-vc` | Path to seed-vc repo |
| `--compile` | off | Use torch.compile for faster inference |
| `--no-vc` | off | TTS-only mode (skip Seed-VC loading) |
| `--port` | 7860 | Gradio server port |

## Testing Without GPU (TTS-only mode)

```bash
python app.py --api-key YOUR_KEY --no-vc
```

This skips Seed-VC loading and returns raw TTS audio instead of voice-converted audio. Useful for testing the UI and TTS pipeline.

## TTS Voices

| Gender | Voice Name | Style |
|--------|-----------|-------|
| Male | Algenib (Chirp 3 HD) | Gravelly, textured |
| Female | Achernar (Chirp 3 HD) | Soft, warm |

## Project Structure

```
voice/
├── app.py                 # Main Gradio application
├── seed_vc_wrapper.py     # Seed-VC v2 inference wrapper
├── tts_generator.py       # Google Cloud TTS generator
├── audio_analysis.py      # Spectrogram & visualization
├── sentences/
│   ├── __init__.py
│   └── data.py            # Preset sentences (EN/ES)
├── requirements.txt
├── setup_server.sh        # Automated server setup
├── .env.example
└── README.md
```

## License

Demo project for educational purposes. Seed-VC is licensed under its own terms — see [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc).
