# 🎯 Accent Trainer — Voice Cloning Demo

AI-powered accent training for language learning. Record your voice, and the system uses zero-shot voice cloning (**Seed-VC v2**, **EZ-VC**, or **OmniVoice**) to generate ideal pronunciation examples **in your own voice** across curated English and Spanish sentences (plus custom user sentences with OmniVoice).

**Languages:** English ↔ Spanish

---

## 🚀 How It Works

```
Your Voice Sample (5-25s) ────────────────────┐
                                              ▼
Pre-generated Reference Audio ────► Seed-VC v2 / EZ-VC / OmniVoice ──► Your Voice Speaking Correctly
 (Native pronunciation/accent)            (Voice Clone)                 (Ideal pronunciation example)
```

1. **Enroll** — Record or upload 5–25 seconds of your natural speech and select a voice cloning backend (`Seed-VC v2`, `EZ-VC`, or `OmniVoice`).
2. **Practice** — Choose a target language (`English` or `Spanish`) and listen to your voice speaking all 10 sentences with native pronunciation (or enter custom sentences when using `OmniVoice`).
3. **Record** — Record or upload yourself attempting each sentence.
4. **Compare** — View side-by-side mel spectrograms, waveforms, and pitch contours (`F0`) of your attempt versus the cloned ideal.

---

## 🛠️ Prerequisites

- **Python 3.10+** (tested on Python 3.10 & 3.11)
- **NVIDIA GPU** (recommended 16GB+ VRAM for real-time inference)
- **FFmpeg** installed on `PATH`

> [!NOTE]
> All 40 reference audio files (10 sentences × 2 languages × 2 genders) are pre-generated and included in `reference_audio/`. **No Google Cloud API keys are required to run the application.**
> To use the **EZ-VC** backend, accept the model license at [`SPRINGLab/EZ-VC`](https://huggingface.co/SPRINGLab/EZ-VC) and set `HF_TOKEN` in `.env` (or via `huggingface-cli login`).

---

## ⚡ GPU Server Setup & Launch

```bash
# 1. Clone the repository
git clone https://github.com/hax2/voicedemo.git
cd voicedemo

# 2. (Optional) Configure .env (e.g., HF_TOKEN for gated EZ-VC weights)
cp .env.example .env

# 3. Run the automated server setup script
# (Clones seed-vc, EZ-VC, and OmniVoice, and installs unified dependencies)
bash setup_server.sh

# 4. Start the application
python app.py
```

---

## 💻 Local Testing (Without GPU)

To test the UI and audio comparison visualizations without a GPU, run in **fallback (`--no-vc`) mode**:

```bash
pip install -r requirements.txt
python app.py --no-vc --no-share
```

---

## 🎛️ Command Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--seed-vc-path` | `./seed-vc` | Path to the cloned `seed-vc` repository |
| `--ez-vc-path` | `./EZ-VC` | Path to the cloned `EZ-VC` repository |
| `--omnivoice-path` | `./OmniVoice` | Path to the cloned `OmniVoice` repository |
| `--compile` | Off | Enables `torch.compile` for Seed-VC v2 AR inference |
| `--no-vc` | Off | Bypasses VC model loading and plays raw native reference audio |
| `--no-share` | Off | Disables Gradio's public `gradio.live` share link |
| `--host` | `0.0.0.0` | Server bind address (use `127.0.0.1` behind a reverse proxy) |
| `--port` | `7860` | Gradio server port |
| `--root-path` | `None` | Public subpath mount point behind Nginx/Caddy (or via `GRADIO_ROOT_PATH`) |

---

## 📂 Project Structure

```
voicedemo/
├── app.py                   # Main Gradio application
├── seed_vc_wrapper.py       # Seed-VC v2 inference & caching wrapper
├── ez_vc_wrapper.py         # EZ-VC (XEUS + F5-TTS) inference & caching wrapper
├── omnivoice_wrapper.py     # OmniVoice zero-shot TTS & voice-cloning wrapper
├── audio_analysis.py        # Spectrogram, waveform, and pitch contour plots
├── generate_references.py   # Utility script used to generate reference_audio/
├── tts_generator.py         # Google Cloud TTS helper module
├── requirements.txt         # Core application dependencies
├── setup_server.sh          # Automated GPU server setup script
├── reference_audio/         # 40 pre-generated reference WAV files
│   ├── en-US/ (male, female)
│   └── es-ES/ (male, female)
└── sentences/
    └── data.py              # Sentence bank & reference path lookup helpers
```

---

## 📜 Credits & License

- **Seed-VC v2** by Plachtaa — [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc)
- **EZ-VC** by SPRING Lab — [EZ-VC/EZ-VC](https://github.com/EZ-VC/EZ-VC)
- **OmniVoice** by Xiaomi / k2-fsa — [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice)
- Reference audio generated with **Google Cloud Text-to-Speech** (Chirp 3 HD: `Algenib` / `Achernar`) and **Edge TTS** fallback.
