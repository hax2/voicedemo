"""Accent Trainer — Seed-VC v2 Gradio Demo

A demo application for accent training in language learning.
Users record their voice, and the system generates ideal pronunciation
audio using their own voice timbre via Seed-VC v2 voice conversion.

Usage:
    python app.py [--api-key YOUR_KEY] [--seed-vc-path ./seed-vc] [--compile]
"""

import argparse
import os
import tempfile
import gradio as gr
import numpy as np
import soundfile as sf

from sentences.data import SENTENCES, VOICE_CONFIG, get_sentences, get_voice_name
from tts_generator import generate_reference_audio
from audio_analysis import create_spectrogram, create_waveform, create_pitch_contour, create_comparison

# ──────────────────────────────────────────────
# Globals
# ──────────────────────────────────────────────
API_KEY = None
vc_wrapper = None


def init_seed_vc(seed_vc_path=None, compile_model=False):
    """Initialize Seed-VC v2 model (called once at startup)."""
    global vc_wrapper
    try:
        from seed_vc_wrapper import get_wrapper
        wrapper = get_wrapper(seed_vc_path=seed_vc_path, compile_model=compile_model)
        wrapper.load()
        vc_wrapper = wrapper
        print("[App] Seed-VC v2 loaded successfully")
    except Exception as e:
        print(f"[App] WARNING: Could not load Seed-VC v2: {e}")
        print("[App] Voice conversion will be unavailable. TTS-only mode active.")
        vc_wrapper = None


# ──────────────────────────────────────────────
# Core pipeline functions
# ──────────────────────────────────────────────
def enroll_voice(audio_path, gender):
    """Process voice enrollment."""
    if audio_path is None:
        return "❌ Please record or upload a voice sample first.", None
    
    # Validate audio length
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=None)
        duration = len(y) / sr
        if duration < 3:
            return f"⚠️ Recording is too short ({duration:.1f}s). Please record at least 5 seconds.", None
        if duration > 30:
            return f"⚠️ Recording is too long ({duration:.1f}s). Please keep it under 30 seconds.", None
        return (
            f"✅ Voice enrolled successfully!\n"
            f"📊 Duration: {duration:.1f}s | Sample rate: {sr}Hz\n"
            f"🗣️ Gender: {gender.title()}\n\n"
            f"Now go to the **Practice** tab to start training!"
        ), audio_path
    except Exception as e:
        return f"❌ Error processing audio: {str(e)}", None


def get_sentence_choices(language):
    """Get sentence choices for the dropdown."""
    sentences = get_sentences(language)
    return [f"{s['text']}  [{s['focus']}]" for s in sentences]


def update_sentences(language):
    """Update sentence dropdown when language changes."""
    choices = get_sentence_choices(language)
    return gr.update(choices=choices, value=choices[0] if choices else None)


def generate_ideal(sentence_choice, language, gender, enrolled_audio):
    """Generate ideal pronunciation using TTS + Seed-VC v2."""
    if not sentence_choice:
        return None, "❌ Please select a sentence first."

    # Extract actual text (remove the focus note in brackets)
    text = sentence_choice.split("  [")[0].strip()

    # Step 1: Generate TTS reference audio
    try:
        status = f"🔊 Generating TTS reference for: \"{text[:60]}...\"\n"
        tts_path = generate_reference_audio(
            text=text,
            language_code=language,
            gender=gender,
            api_key=API_KEY,
        )
        status += "✅ TTS reference generated\n"
    except Exception as e:
        return None, f"❌ TTS generation failed: {str(e)}"

    # Step 2: Voice conversion with Seed-VC v2
    if vc_wrapper is not None and enrolled_audio is not None:
        try:
            status += "🎙️ Converting to your voice with Seed-VC v2...\n"
            # source = TTS audio (correct pronunciation)
            # reference = user's voice (timbre to clone)
            converted_path = vc_wrapper.convert_voice(
                source_audio_path=tts_path,
                reference_audio_path=enrolled_audio,
                diffusion_steps=25,
                length_adjust=1.0,
                inference_cfg_rate=0.7,
            )
            status += "✅ Voice conversion complete! Listen to your ideal pronunciation below."
            return converted_path, status
        except Exception as e:
            status += f"⚠️ Voice conversion failed ({str(e)}). Returning TTS audio instead.\n"
            return tts_path, status
    else:
        if enrolled_audio is None:
            status += "⚠️ No voice enrolled — returning raw TTS audio. Enroll your voice for personalized output.\n"
        else:
            status += "⚠️ Seed-VC not loaded — returning raw TTS audio.\n"
        return tts_path, status


def run_comparison(ideal_audio, attempt_audio):
    """Generate comparison visualizations."""
    if ideal_audio is None and attempt_audio is None:
        return (
            create_comparison(None, None),
            "❌ Please generate ideal audio and record an attempt first."
        )

    fig = create_comparison(ideal_audio, attempt_audio)
    
    notes = []
    if ideal_audio is None:
        notes.append("⚠️ No ideal audio — generate it first")
    if attempt_audio is None:
        notes.append("⚠️ No attempt recorded — record yourself speaking the sentence")
    
    if not notes:
        notes.append("✅ Comparison generated! Look at the spectrograms to compare your pronunciation patterns.")
        notes.append("💡 **Tip:** Similar spectrogram shapes = similar pronunciation. Pay attention to pitch contours and frequency patterns.")
    
    return fig, "\n".join(notes)


# ──────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────
def build_ui():
    """Build the Gradio Blocks UI."""

    # CSS for premium dark theme
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #e94560 0%, #0f3460 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1em;
        margin-top: 0;
    }
    .tab-content {
        padding: 20px 10px;
    }
    """

    with gr.Blocks(
        title="Accent Trainer — Seed-VC v2",
        theme=gr.themes.Soft(
            primary_hue="red",
            secondary_hue="blue",
            neutral_hue="slate",
        ),
        css=custom_css,
    ) as demo:

        # ── Header ──
        gr.HTML("""
            <div style="text-align: center; padding: 20px 0 10px;">
                <h1 class="main-title">🎯 Accent Trainer</h1>
                <p class="subtitle">Perfect your pronunciation with AI voice cloning — powered by Seed-VC v2</p>
            </div>
        """)

        # ── State ──
        enrolled_audio_state = gr.State(None)
        enrolled_gender_state = gr.State("male")
        ideal_audio_state = gr.State(None)

        with gr.Tabs():
            # ═══════════════════════════════════════════
            # TAB 1: Voice Enrollment
            # ═══════════════════════════════════════════
            with gr.Tab("🎤 Voice Enrollment", id="enrollment"):
                gr.Markdown("""
                ### Step 1: Enroll Your Voice
                Record or upload a **10–25 second** sample of your natural speech.  
                This will be used to clone your voice timbre for ideal pronunciation examples.
                
                **Tips:** Speak clearly in your native language. Read a paragraph from a book, 
                describe your day, or count to twenty — anything that captures your natural voice.
                """)

                with gr.Row():
                    with gr.Column(scale=2):
                        enrollment_audio = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="🎙️ Your Voice Sample",
                            max_length=30,
                        )
                    with gr.Column(scale=1):
                        gender_select = gr.Radio(
                            choices=["male", "female"],
                            value="male",
                            label="🗣️ Voice Type",
                            info="Selects the TTS base voice (Algenib / Achernar)",
                        )

                enroll_btn = gr.Button("✅ Enroll Voice", variant="primary", size="lg")
                enrollment_status = gr.Textbox(
                    label="Status", interactive=False, lines=4
                )

                enroll_btn.click(
                    fn=enroll_voice,
                    inputs=[enrollment_audio, gender_select],
                    outputs=[enrollment_status, enrolled_audio_state],
                )
                gender_select.change(
                    fn=lambda g: g,
                    inputs=[gender_select],
                    outputs=[enrolled_gender_state],
                )

            # ═══════════════════════════════════════════
            # TAB 2: Practice
            # ═══════════════════════════════════════════
            with gr.Tab("📝 Practice", id="practice"):
                gr.Markdown("""
                ### Step 2: Practice Pronunciation
                Select a target language and sentence, then generate the ideal pronunciation in your voice.  
                Record yourself attempting the sentence and compare!
                """)

                with gr.Row():
                    language_select = gr.Dropdown(
                        choices=[
                            ("English (for Spanish speakers)", "en-US"),
                            ("Spanish (for English speakers)", "es-ES"),
                        ],
                        value="en-US",
                        label="🌍 Target Language",
                    )

                # Initialize sentence list
                initial_sentences = get_sentence_choices("en-US")
                sentence_select = gr.Dropdown(
                    choices=initial_sentences,
                    value=initial_sentences[0] if initial_sentences else None,
                    label="📄 Select a Sentence",
                    info="Each sentence targets specific pronunciation challenges",
                )

                language_select.change(
                    fn=update_sentences,
                    inputs=[language_select],
                    outputs=[sentence_select],
                )

                generate_btn = gr.Button(
                    "🔊 Generate Ideal Pronunciation",
                    variant="primary",
                    size="lg",
                )
                generation_status = gr.Textbox(
                    label="Generation Status", interactive=False, lines=4
                )

                gr.Markdown("---")

                with gr.Row(equal_height=True):
                    with gr.Column():
                        gr.Markdown("#### 🎯 Ideal Pronunciation")
                        ideal_audio = gr.Audio(
                            label="How it should sound (in your voice)",
                            type="filepath",
                            interactive=False,
                        )
                    with gr.Column():
                        gr.Markdown("#### 🎤 Your Attempt")
                        attempt_audio = gr.Audio(
                            sources=["microphone"],
                            type="filepath",
                            label="Record yourself saying the sentence",
                        )

                generate_btn.click(
                    fn=generate_ideal,
                    inputs=[sentence_select, language_select, enrolled_gender_state, enrolled_audio_state],
                    outputs=[ideal_audio, generation_status],
                ).then(
                    fn=lambda x: x,
                    inputs=[ideal_audio],
                    outputs=[ideal_audio_state],
                )

                gr.Markdown("---")

                compare_btn = gr.Button(
                    "📊 Compare Pronunciation",
                    variant="secondary",
                    size="lg",
                )
                comparison_status = gr.Textbox(
                    label="Comparison Notes", interactive=False, lines=3
                )
                comparison_plot = gr.Plot(
                    label="Pronunciation Comparison",
                )

                compare_btn.click(
                    fn=run_comparison,
                    inputs=[ideal_audio, attempt_audio],
                    outputs=[comparison_plot, comparison_status],
                )

            # ═══════════════════════════════════════════
            # TAB 3: Visualizations
            # ═══════════════════════════════════════════
            with gr.Tab("📊 Analysis", id="analysis"):
                gr.Markdown("""
                ### Detailed Audio Analysis
                Upload or use audio from the Practice tab for detailed spectrogram analysis.
                """)

                with gr.Row():
                    analysis_audio = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="🎵 Audio to Analyze",
                    )

                analyze_btn = gr.Button("🔍 Analyze Audio", variant="primary")

                with gr.Row():
                    spec_plot = gr.Plot(label="Mel Spectrogram")
                with gr.Row():
                    wave_plot = gr.Plot(label="Waveform")
                with gr.Row():
                    pitch_plot = gr.Plot(label="Pitch Contour")

                def analyze_audio(audio_path):
                    return (
                        create_spectrogram(audio_path, "Mel Spectrogram"),
                        create_waveform(audio_path, "Waveform"),
                        create_pitch_contour(audio_path, "Pitch Contour (F0)"),
                    )

                analyze_btn.click(
                    fn=analyze_audio,
                    inputs=[analysis_audio],
                    outputs=[spec_plot, wave_plot, pitch_plot],
                )

            # ═══════════════════════════════════════════
            # TAB 4: About
            # ═══════════════════════════════════════════
            with gr.Tab("ℹ️ About", id="about"):
                gr.Markdown("""
                ### How It Works
                
                This accent training demo uses a three-stage pipeline:
                
                1. **🗣️ Voice Enrollment** — You record a short sample of your natural speech.
                   This captures your unique voice characteristics (timbre, tone, resonance).
                
                2. **🔊 TTS Reference Generation** — Google Cloud TTS (Chirp 3 HD voices)
                   generates a perfect pronunciation of the target sentence in the chosen language.
                
                3. **🎯 Voice Conversion (Seed-VC v2)** — The TTS audio is converted to sound like
                   *your* voice using zero-shot voice conversion. The result preserves the correct
                   pronunciation and accent from the TTS, but in your voice timbre.
                
                4. **📊 Comparison** — Record yourself attempting the sentence, then compare
                   spectrograms, waveforms, and pitch contours side by side.
                
                ---
                
                ### Technology
                
                | Component | Technology |
                |-----------|-----------|
                | Voice Conversion | [Seed-VC v2](https://github.com/Plachtaa/seed-vc) — zero-shot diffusion-based VC |
                | Text-to-Speech | Google Cloud TTS — Chirp 3 HD (Algenib / Achernar) |
                | Visualization | Librosa + Matplotlib — mel spectrograms, pitch tracking |
                | Interface | Gradio Blocks — with `share=True` for remote access |
                
                ---
                
                ### Languages
                
                Currently supported in this demo:
                - **English** (for Spanish speakers) — focuses on TH, SH, V/B, schwa sounds
                - **Spanish** (for English speakers) — focuses on rolled R, vowel purity, J sound
                """)

        # ── Footer ──
        gr.HTML("""
            <div style="text-align: center; padding: 20px; color: #666; font-size: 0.9em;">
                Accent Trainer Demo — Built with Seed-VC v2 & Google Cloud TTS
            </div>
        """)

    return demo


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Accent Trainer Demo")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Google Cloud TTS API key")
    parser.add_argument("--seed-vc-path", type=str, default=None,
                        help="Path to cloned seed-vc repo")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for faster inference")
    parser.add_argument("--no-vc", action="store_true",
                        help="Skip loading Seed-VC (TTS-only mode for testing)")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port for Gradio server")
    args = parser.parse_args()

    # Set API key
    global API_KEY
    API_KEY = args.api_key or os.environ.get("GOOGLE_TTS_API_KEY", "AIzaSyD1DrcOjKZ_eEF9oBe1ZyFHVhZKIUKuWc4")

    if not API_KEY:
        print("[App] WARNING: No Google TTS API key provided. Set --api-key or GOOGLE_TTS_API_KEY env var.")

    # Load Seed-VC v2
    if not args.no_vc:
        init_seed_vc(seed_vc_path=args.seed_vc_path, compile_model=args.compile)
    else:
        print("[App] Running in TTS-only mode (--no-vc flag set)")

    # Build and launch Gradio app
    demo = build_ui()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=args.port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
