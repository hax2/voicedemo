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
import torch

from sentences.data import SENTENCES, get_sentences, get_reference_audio_path
from audio_analysis import create_spectrogram, create_waveform, create_pitch_contour, create_comparison

# ──────────────────────────────────────────────
# Globals
# ──────────────────────────────────────────────
vc_wrappers = {
    "Seed-VC v2": None,
    "EZ-VC": None,
    "OmniVoice": None
}

def init_omnivoice():
    """Initialize OmniVoice model."""
    try:
        from omnivoice_wrapper import get_wrapper
        wrapper = get_wrapper()
        vc_wrappers["OmniVoice"] = wrapper
        print("[App] OmniVoice wrapper initialized")
    except Exception as e:
        print(f"[App] WARNING: Could not load OmniVoice: {e}")
        vc_wrappers["OmniVoice"] = None

def init_seed_vc(seed_vc_path=None, compile_model=False):
    """Initialize Seed-VC v2 model (called once at startup)."""
    try:
        from seed_vc_wrapper import get_wrapper
        wrapper = get_wrapper(seed_vc_path=seed_vc_path, compile_model=compile_model)
        vc_wrappers["Seed-VC v2"] = wrapper
        print("[App] Seed-VC v2 wrapper initialized")
    except Exception as e:
        print(f"[App] WARNING: Could not load Seed-VC v2: {e}")
        vc_wrappers["Seed-VC v2"] = None

def init_ez_vc(ez_vc_path=None):
    """Initialize EZ-VC model (called once at startup)."""
    try:
        from ez_vc_wrapper import get_wrapper
        wrapper = get_wrapper(ez_vc_path=ez_vc_path)
        vc_wrappers["EZ-VC"] = wrapper
        print("[App] EZ-VC wrapper initialized")
    except Exception as e:
        print(f"[App] WARNING: Could not load EZ-VC: {e}")
        vc_wrappers["EZ-VC"] = None


# ──────────────────────────────────────────────
# Core pipeline functions
# ──────────────────────────────────────────────
def enroll_voice(audio_path, gender, vc_model, cfg_rate, progress=gr.Progress()):
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

        if vc_model not in vc_wrappers or vc_wrappers[vc_model] is None:
            return "❌ Selected VC model is not available. Please check the logs.", None

        vc_wrapper = vc_wrappers[vc_model]

        # Ensure selected model is loaded
        if hasattr(vc_wrapper, "load"):
            vc_wrapper.load()

        # Check if selected VC model is loaded for bulk generation
        wrapper = vc_wrappers.get(vc_model)
        if wrapper is not None:
            if vc_model == "OmniVoice":
                status += f"🎙️ {vc_model} active. Registration complete! You can now use Custom TTS in the Practice tab.\n"
                return status, audio_path
                
            status += f"🎙️ {vc_model} active. Bulk-generating all sentences in your voice...\n"
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
                        if vc_model == "OmniVoice":
                            wrapper.generate_tts(
                                text=text,
                                reference_audio_path=audio_path,
                                use_cache=True
                            )
                        else:
                            wrapper.convert_voice(
                                source_audio_path=ref_path,
                                reference_audio_path=audio_path,
                                diffusion_steps=25,
                                length_adjust=1.0,
                                inference_cfg_rate=cfg_rate,
                            )
                    except Exception as e:
                        print(f"[App] Error bulk-converting sentence {i} in {lang}: {e}")
                
                progress(1.0, desc="All conversions complete!")
                status += f"⚡ Bulk-generation complete! All {total} sentences are pre-converted and will play instantly in the Practice tab."
            else:
                status += "⚠️ No reference audio files found to convert."
        else:
            status += f"⚠️ {vc_model} not loaded. Voice conversion is offline (playing raw reference audio)."

        return status, audio_path
    except Exception as e:
        return f"❌ Error processing voice enrollment: {str(e)}", None


def update_practice_sentences(language, gender, enrolled_audio, vc_model, cfg_rate):
    """Update all 10 sentence cards in the Practice tab."""
    sentences = get_sentences(language)
    outputs = []
    
    wrapper = vc_wrappers.get(vc_model)
    import hashlib
    import os
    from seed_vc_wrapper import CACHE_DIR
    
    for i in range(10):
        sentence_html_val = ""
        ref_path = None
        ideal_path = None
        visible = (i < len(sentences))
        
        if visible:
            s = sentences[i]
            text = s["text"]
            focus = s["focus"]
            diff = "Beginner" if i < 5 else "Intermediate"
            sentence_html_val = f"""
            <div style="padding: 10px 15px; border-radius: 6px; background: #1a1a2e; border: 1px solid #162447;">
                <p style="color: #e94560; font-size: 0.8em; font-weight: bold; text-transform: uppercase; margin: 0 0 3px 0; letter-spacing: 0.5px;">Sentence #{i+1} — {diff}</p>
                <h3 style="font-size: 1.3em; margin: 0 0 5px 0; color: #ffffff; line-height: 1.2;">{text}</h3>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="background: #0f3460; color: #e94560; font-size: 0.75em; padding: 2px 6px; border-radius: 4px; font-weight: bold;">Focus</span>
                    <p style="color: #c5c5c5; margin: 0; font-size: 0.85em;">{focus}</p>
                </div>
            </div>
            """
            ref_path = get_reference_audio_path(language, gender, i)
            
            if wrapper is not None and enrolled_audio is not None and ref_path is not None:
                try:
                    if vc_model == "OmniVoice":
                        ideal_path = wrapper.generate_tts(
                            text=text,
                            reference_audio_path=enrolled_audio,
                            use_cache=True
                        )
                    else:
                        ideal_path = wrapper.convert_voice(
                            source_audio_path=ref_path,
                            reference_audio_path=enrolled_audio,
                            diffusion_steps=25,
                            length_adjust=1.0,
                            inference_cfg_rate=cfg_rate,
                            use_cache=True
                        )
                except Exception as e:
                    print(f"[App] Error generating ideal voice for sentence {i}: {e}")
                    ideal_path = ref_path

            else:
                ideal_path = ref_path
        
        # We append: sentence_html, native_player, cloned_player, attempt_recorder, comparison_plot, comparison_status
        outputs.append(gr.update(value=sentence_html_val, visible=visible))
        outputs.append(gr.update(value=ref_path, visible=visible))
        outputs.append(gr.update(value=ideal_path, visible=visible))
        outputs.append(gr.update(value=None, visible=visible))  # Clear attempt recording
        outputs.append(gr.update(value=None, visible=visible))  # Clear comparison plot
        outputs.append(gr.update(value="", visible=visible))    # Clear comparison status
        
    return outputs


def generate_custom_tts(text, enrolled_audio, vc_model):
    """Generate custom TTS using OmniVoice."""
    if vc_model != "OmniVoice":
        return None
    
    if not text or not text.strip():
        return None
        
    wrapper = vc_wrappers.get(vc_model)
    if wrapper is None or enrolled_audio is None:
        return None

    try:
        ideal_path = wrapper.generate_tts(
            text=text,
            reference_audio_path=enrolled_audio,
            use_cache=False
        )
        return ideal_path
    except Exception as e:
        print(f"[App] Error generating custom TTS: {e}")
        return None


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
        vc_model_state = gr.State("Seed-VC v2")
        cfg_rate_state = gr.State(2.0)

        with gr.Tabs():
            # ═══════════════════════════════════════════
            # TAB 1: Voice Enrollment
            # ═══════════════════════════════════════════
            with gr.Tab("🎤 Voice Enrollment", id="enrollment"):
                gr.Markdown("""
                ### Step 1: Enroll Your Voice
                Record or upload a **10–25 second** sample of your natural speech.  
                This will be used to clone your voice timbre for ideal pronunciation examples.
                
                **Important:** Please read one of the following phrases for your recording:
                - *English:* "Hello, this is my voice and I am reading this sentence to test the system."
                - *Spanish:* "Hola, esta es mi voz y estoy leyendo esta frase para probar el sistema."
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
                            info="Selects the TTS base voice",
                        )
                        vc_model_select = gr.Radio(
                            choices=["Seed-VC v2", "EZ-VC", "OmniVoice"],
                            value="Seed-VC v2",
                            label="⚙️ VC Model",
                            info="Choose the voice conversion backend"
                        )
                        cfg_rate_slider = gr.Slider(
                            minimum=0.5, maximum=3.0, value=2.0, step=0.1,
                            label="🎚️ CFG Rate",
                            info="Conditioning strength. Higher = sticks more to your voice."
                        )

                enroll_btn = gr.Button("✅ Enroll Voice", variant="primary", size="lg")
                enrollment_status = gr.Textbox(
                    label="Status", interactive=False, lines=4
                )

            # ═══════════════════════════════════════════
            # TAB 2: Practice
            # ═══════════════════════════════════════════
            with gr.Tab("📝 Practice", id="practice"):
                gr.Markdown("""
                ### Step 2: Practice Pronunciation
                Choose your target language. Below is the list of target sentences.
                Listen to the native reference, compare it to your cloned "ideal" voice, record your attempt, and run the comparison analysis!
                """)

                # Standard Practice Group (Visible when NOT OmniVoice)
                with gr.Group(visible=True) as standard_practice_group:
                    with gr.Row():
                        language_select = gr.Radio(
                            choices=[
                                ("English (for Spanish speakers)", "en-US"),
                                ("Spanish (for English speakers)", "es-ES"),
                            ],
                            value="en-US",
                            label="🌍 Target Language",
                        )

                    # Dynamically build 10 sentence cards
                    sentence_htmls = []
                    native_players = []
                    cloned_players = []
                    attempt_recorders = []
                    compare_btns = []
                    comparison_statuses = []
                    comparison_plots = []

                    for i in range(10):
                        with gr.Group():
                            s_html = gr.HTML(value="")
                            sentence_htmls.append(s_html)

                            with gr.Row():
                                with gr.Column(scale=1):
                                    ref_player = gr.Audio(
                                        label="📖 Native Reference",
                                        type="filepath",
                                        interactive=False,
                                    )
                                    native_players.append(ref_player)
                                with gr.Column(scale=1):
                                    clone_player = gr.Audio(
                                        label="🎯 Cloned Ideal (Your Voice)",
                                        type="filepath",
                                        interactive=False,
                                    )
                                    cloned_players.append(clone_player)
                                with gr.Column(scale=1):
                                    attempt_player = gr.Audio(
                                        sources=["microphone"],
                                        type="filepath",
                                        label="🎤 Your Attempt",
                                    )
                                    attempt_recorders.append(attempt_player)

                            with gr.Accordion("📊 Comparison Analysis", open=False):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        comp_btn = gr.Button("📊 Analyze & Compare", variant="secondary", size="sm")
                                        compare_btns.append(comp_btn)
                                        comp_status = gr.Textbox(
                                            label="Comparison Notes",
                                            interactive=False,
                                            lines=3,
                                        )
                                        comparison_statuses.append(comp_status)
                                    with gr.Column(scale=2):
                                        comp_plot = gr.Plot(label="Spectrogram / Pitch Comparison")
                                        comparison_plots.append(comp_plot)

                            gr.HTML("<div style='margin-bottom: 25px;'></div>")

                # Custom TTS (Visible only when OmniVoice is active)
                with gr.Group(visible=False) as custom_tts_group:
                    gr.Markdown("### Custom Sentence Generation (OmniVoice Only)")
                    with gr.Row():
                        with gr.Column(scale=3):
                            custom_tts_text = gr.Textbox(label="Enter your own sentence to practice", placeholder="Type a sentence here...")
                        with gr.Column(scale=1):
                            custom_tts_btn = gr.Button("🗣️ Generate TTS")
                    custom_tts_audio = gr.Audio(label="Custom Cloned Audio", interactive=False)

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

        # ──────────────────────────────────────────────
        # Event Listeners & Event Bindings
        # ──────────────────────────────────────────────
        
        # Collect outputs for practice tab sentence cards
        practice_outputs = []
        for i in range(10):
            practice_outputs.extend([
                sentence_htmls[i],
                native_players[i],
                cloned_players[i],
                attempt_recorders[i],
                comparison_plots[i],
                comparison_statuses[i],
            ])

        # TAB 1: Voice Enrollment
        enroll_btn.click(
            fn=enroll_voice,
            inputs=[enrollment_audio, gender_select, vc_model_select, cfg_rate_slider],
            outputs=[enrollment_status, enrolled_audio_state],
        ).then(
            fn=update_practice_sentences,
            inputs=[language_select, enrolled_gender_state, enrolled_audio_state, vc_model_state, cfg_rate_state],
            outputs=practice_outputs,
        )

        gender_select.change(
            fn=lambda g: g,
            inputs=[gender_select],
            outputs=[enrolled_gender_state],
        )

        def toggle_custom_tts(model_name):
            return gr.update(visible=(model_name == "OmniVoice")), gr.update(visible=(model_name != "OmniVoice"))

        vc_model_select.change(
            fn=lambda m: m,
            inputs=[vc_model_select],
            outputs=[vc_model_state],
        ).then(
            fn=toggle_custom_tts,
            inputs=[vc_model_select],
            outputs=[custom_tts_group, standard_practice_group],
        ).then(
            fn=update_practice_sentences,
            inputs=[language_select, enrolled_gender_state, enrolled_audio_state, vc_model_state, cfg_rate_state],
            outputs=practice_outputs,
        )

        custom_tts_btn.click(
            fn=generate_custom_tts,
            inputs=[custom_tts_text, enrolled_audio_state, vc_model_state],
            outputs=[custom_tts_audio]
        )

        cfg_rate_slider.change(
            fn=lambda c: c,
            inputs=[cfg_rate_slider],
            outputs=[cfg_rate_state],
        ).then(
            fn=update_practice_sentences,
            inputs=[language_select, enrolled_gender_state, enrolled_audio_state, vc_model_state, cfg_rate_state],
            outputs=practice_outputs,
        )

        # TAB 2: Practice (Sentence events: language changes and comparison clicks)
        language_select.change(
            fn=update_practice_sentences,
            inputs=[
                language_select,
                enrolled_gender_state,
                enrolled_audio_state,
                vc_model_state,
                cfg_rate_state,
            ],
            outputs=practice_outputs,
        )

        for i in range(10):
            compare_btns[i].click(
                fn=run_comparison,
                inputs=[cloned_players[i], attempt_recorders[i]],
                outputs=[comparison_plots[i], comparison_statuses[i]],
            )

        # TAB 3: Analysis
        analyze_btn.click(
            fn=analyze_audio,
            inputs=[analysis_audio],
            outputs=[spec_plot, wave_plot, pitch_plot],
        )

        # Trigger initial load on app startup
        demo.load(
            fn=update_practice_sentences,
            inputs=[
                language_select,
                enrolled_gender_state,
                enrolled_audio_state,
                vc_model_state,
                cfg_rate_state,
            ],
            outputs=practice_outputs,
        )

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
    parser.add_argument("--ez-vc-path", type=str, default=None,
                        help="Path to cloned EZ-VC repo")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for faster inference")
    parser.add_argument("--no-vc", action="store_true",
                        help="Skip loading Seed-VC (reference-audio-only mode for testing)")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port for Gradio server")
    args = parser.parse_args()

    # Load VC models
    if not args.no_vc:
        init_seed_vc(seed_vc_path=args.seed_vc_path, compile_model=args.compile)
        init_ez_vc(ez_vc_path=args.ez_vc_path)
        init_omnivoice()
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
