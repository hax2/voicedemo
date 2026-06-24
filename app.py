"""Accent Trainer — Seed-VC v2 Gradio Demo

A demo application for accent training in language learning.
Users record their voice, and the system uses Seed-VC v2 to convert
pre-generated reference audio (correct pronunciation) into the user's
voice timbre. Users then practice and compare via spectrograms.

Reference audio is pre-generated once via generate_references.py and
shipped as static assets in reference_audio/.

Usage:
    python app.py [--seed-vc-path ./seed-vc] [--compile] [--no-vc]
"""

import argparse
import os
import gradio as gr

from sentences.data import SENTENCES, get_sentences, get_reference_audio_path
from audio_analysis import create_spectrogram, create_waveform, create_pitch_contour, create_comparison

# ──────────────────────────────────────────────
# Globals
# ──────────────────────────────────────────────
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
        print("[App] Voice conversion will be unavailable. Reference-audio-only mode active.")
        vc_wrapper = None


# ──────────────────────────────────────────────
# Core pipeline functions
# ──────────────────────────────────────────────
def enroll_voice(audio_path, gender, progress=gr.Progress()):
    """Process voice enrollment and bulk-generate all target sentences."""
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

        status = (
            f"✅ Voice enrolled successfully!\n"
            f"📊 Duration: {duration:.1f}s | Sample rate: {sr}Hz\n"
            f"🗣️ Gender: {gender.title()}\n\n"
        )

        # Check if Seed-VC is loaded for bulk generation
        global vc_wrapper
        if vc_wrapper is not None:
            status += "🎙️ Seed-VC v2 active. Bulk-generating all sentences in your voice...\n"
            print("[App] Starting bulk voice conversion for enrolled voice...")

            # List of all sentences to convert
            sentences_to_convert = []
            for lang in ["en-US", "es-ES"]:
                lang_sentences = get_sentences(lang)
                for i, s in enumerate(lang_sentences):
                    ref_path = get_reference_audio_path(lang, gender, i)
                    if ref_path and os.path.exists(ref_path):
                        sentences_to_convert.append((lang, i, ref_path, s["text"]))

            total = len(sentences_to_convert)
            if total > 0:
                progress(0, desc="Starting voice conversion...")
                for idx, (lang, i, ref_path, text) in enumerate(sentences_to_convert):
                    desc = f"Converting ({lang}) {idx+1}/{total}: {text[:25]}..."
                    progress(idx / total, desc=desc)
                    try:
                        # This runs conversion and caches the output path
                        vc_wrapper.convert_voice(
                            source_audio_path=ref_path,
                            reference_audio_path=audio_path,
                            diffusion_steps=25,
                            length_adjust=1.0,
                            inference_cfg_rate=0.7,
                        )
                    except Exception as e:
                        print(f"[App] Error bulk-converting sentence {i} in {lang}: {e}")
                
                progress(1.0, desc="All conversions complete!")
                status += f"⚡ Bulk-generation complete! All {total} sentences are pre-converted and will play instantly in the Practice tab."
            else:
                status += "⚠️ No reference audio files found to convert."
        else:
            status += "⚠️ Seed-VC not loaded. Voice conversion is offline (playing raw reference audio)."

        return status, audio_path
    except Exception as e:
        return f"❌ Error processing voice enrollment: {str(e)}", None


def get_sentence_choices(language):
    """Get sentence choices for the dropdown."""
    sentences = get_sentences(language)
    return [f"{i}|{s['text']}  [{s['focus']}]" for i, s in enumerate(sentences)]


def update_sentences(language):
    """Update sentence dropdown when language changes."""
    choices = get_sentence_choices(language)
    return gr.update(choices=choices, value=choices[0] if choices else None)


def generate_ideal(sentence_choice, language, gender, enrolled_audio):
    """Generate ideal pronunciation by converting pre-generated reference audio
    through Seed-VC v2 using the user's enrolled voice."""
    if not sentence_choice:
        return None, None, "❌ Please select a sentence first."

    # Parse sentence index from choice string ("0|text  [focus]")
    sentence_index = int(sentence_choice.split("|")[0])

    # Step 1: Look up the pre-generated reference audio
    ref_path = get_reference_audio_path(language, gender, sentence_index)
    if ref_path is None:
        return None, None, (
            f"❌ Reference audio not found for sentence {sentence_index + 1}.\n"
            f"Run `python generate_references.py` first to create the audio files."
        )

    status = f"🔊 Loaded reference audio: sentence {sentence_index + 1}\n"

    # Step 2: Voice conversion with Seed-VC v2
    if vc_wrapper is not None and enrolled_audio is not None:
        try:
            status += "🎙️ Converting to your voice with Seed-VC v2...\n"
            # source = pre-generated TTS audio (correct pronunciation)
            # reference = user's enrolled voice (timbre to clone)
            converted_path = vc_wrapper.convert_voice(
                source_audio_path=ref_path,
                reference_audio_path=enrolled_audio,
                diffusion_steps=25,
                length_adjust=1.0,
                inference_cfg_rate=0.7,
            )
            status += "✅ Voice conversion complete! Listen to your ideal pronunciation below."
            return converted_path, ref_path, status
        except Exception as e:
            status += f"⚠️ Voice conversion failed ({str(e)}). Playing raw reference audio instead.\n"
            return ref_path, ref_path, status
    else:
        if enrolled_audio is None:
            status += "⚠️ No voice enrolled — playing raw reference audio. Enroll your voice for a personalized version.\n"
        else:
            status += "⚠️ Seed-VC not loaded — playing raw reference audio.\n"
        return ref_path, ref_path, status


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
                Select a target language and sentence. The system will convert the reference 
                pronunciation into your voice using Seed-VC v2, so you can hear how *you* should sound.
                Then record yourself and compare!
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
                    "🔊 Generate Ideal Pronunciation (in Your Voice)",
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
                        gr.Markdown("#### 📖 Original Reference")
                        reference_audio_player = gr.Audio(
                            label="Original TTS reference (native speaker)",
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
                    outputs=[ideal_audio, reference_audio_player, generation_status],
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
                Upload or record audio for detailed spectrogram analysis.
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
                
                1. **🗣️ Voice Enrollment** — You record a short sample of your natural speech.
                
                2. **🔊 Pre-generated References** — 10 sentences per language have been 
                   pre-recorded using Google Cloud TTS (Chirp 3 HD voices). These are static 
                   assets — no API calls happen at runtime.
                
                3. **🎯 Voice Conversion (Seed-VC v2)** — When you select a sentence, the 
                   pre-generated reference audio is converted to sound like *your* voice using 
                   zero-shot voice conversion. Correct pronunciation + your timbre.
                
                4. **📊 Comparison** — Record yourself attempting the sentence, then compare
                   spectrograms, waveforms, and pitch contours side by side.
                
                ---
                
                ### Technology
                
                | Component | Technology |
                |-----------|-----------|
                | Voice Conversion | [Seed-VC v2](https://github.com/Plachtaa/seed-vc) — zero-shot diffusion-based VC |
                | Reference Audio | Google Cloud TTS — Chirp 3 HD (Algenib / Achernar) |
                | Visualization | Librosa + Matplotlib — mel spectrograms, pitch tracking |
                | Interface | Gradio Blocks — with `share=True` for remote access |
                
                ---
                
                ### Languages
                
                - **English** (for Spanish speakers) — TH, SH, V/B, schwa, consonant clusters
                - **Spanish** (for English speakers) — rolled R, vowel purity, J sound, -ción endings
                
                10 sentences per language × 2 genders = 40 pre-generated reference audio files.
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
    parser.add_argument("--seed-vc-path", type=str, default=None,
                        help="Path to cloned seed-vc repo")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for faster inference")
    parser.add_argument("--no-vc", action="store_true",
                        help="Skip loading Seed-VC (reference-audio-only mode for testing)")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port for Gradio server")
    args = parser.parse_args()

    # Load Seed-VC v2
    if not args.no_vc:
        init_seed_vc(seed_vc_path=args.seed_vc_path, compile_model=args.compile)
    else:
        print("[App] Running in reference-audio-only mode (--no-vc flag set)")

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
