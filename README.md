# 🎯 Accent Trainer — Seed-VC v2 Demo

AI-powered accent training for language learning. Record your voice, and the system uses zero-shot voice conversion via **Seed-VC v2** to generate ideal pronunciation examples **in your own voice** from a set of pre-selected sentences.

**Languages:** English ↔ Spanish

---

## 🚀 How It Works

```
Your Voice Sample (10-25s) ──────────────────┐
                                            ▼
Pre-generated Reference Audio ────────► Seed-VC v2 ──► Your Voice Speaking Correctly
 (Perfect pronunciation/accent)        (Voice Clone)    (Ideal pronunciation example)
```

1. **Enroll** — Record 10-25 seconds of your natural voice speaking in your native language.
2. **Practice** — Choose a target sentence. The system uses **Seed-VC v2** to convert the pre-recorded ideal pronunciation of that sentence into your own voice.
3. **Record** — Record yourself attempting to speak the sentence.
4. **Compare** — View side-by-side pitch contours, waveforms, and mel spectrograms of your attempt versus the ideal clone to identify visual and auditory accent gaps.

---

## 🛠️ Prerequisites

- **Python 3.10**
- **NVIDIA GPU** (recommended A100 40GB or similar for real-time inference)
- **FFmpeg** installed on the system path

> [!NOTE]
> All reference audio files (10 sentences × 2 languages × 2 genders = 40 total files) are already pre-generated and included in this repository. **No Google Cloud API keys or external network requests are needed to run this application!**

---

## ⚡ Deployment on A100 Server

Follow these steps to deploy and run the demo on your GPU server:

```bash
# 1. Clone the repository
git clone https://github.com/hax2/voicedemo.git
cd voicedemo

# 2. Run the automated server setup script
# (Clones seed-vc, downloads checkpoints, and installs all dependencies)
bash setup_server.sh

# 3. Start the application
python app.py
```

Once running, Gradio will output a public URL (e.g., `https://xxxx.gradio.live`) that you can open on your laptop's browser to access the demo.

---

## 💻 Local Testing (Without GPU)

If you want to test the UI locally on your laptop without an NVIDIA GPU, you can run the app in **TTS-only / no-VC fallback mode**:

```bash
# Install dependencies
pip install -r requirements.txt

# Run in no-VC mode
python app.py --no-vc
```
*Note: In `--no-vc` mode, the app will play the original pre-generated reference voices instead of converting them to your voice.*

---

## 🎛️ Command Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--seed-vc-path` | `./seed-vc` | Path to the cloned `seed-vc` folder |
| `--compile` | Off | Enables `torch.compile` for faster inference (requires PyTorch 2.x) |
| `--no-vc` | Off | Bypasses Seed-VC model loading (runs in fallback mode) |
| `--port` | `7860` | Gradio server port |

---

## 📂 Project Structure

```
voicedemo/
├── app.py                   # Main Gradio application
├── seed_vc_wrapper.py       # Seed-VC v2 model inference & caching wrapper
├── audio_analysis.py        # Visualizations (spectrogram, pitch contour)
├── generate_references.py   # Script used to pre-generate the reference audio
├── requirements.txt         # Core dependencies
├── setup_server.sh          # Automated server setup and model downloader
├── reference_audio/         # 40 pre-generated reference WAV files (static assets)
│   ├── en-US/
│   │   ├── male/
│   │   └── female/
│   └── es-ES/
│       ├── male/
│       └── female/
└── sentences/
    └── data.py              # Sentence data & path lookup helper
```

---

## 📜 Credits & License

- **Seed-VC v2** by Plachtaa. See [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc) for model details, code, and license.
- Audio references were generated using **Google Cloud Text-to-Speech** (Chirp 3 HD: `Algenib` for male, `Achernar` for female) and **Edge TTS** fallback.
