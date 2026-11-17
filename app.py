"""Accent Trainer — Multi-Backend Voice Cloning Gradio Demo

A demo application for accent training in language learning.
Users record their voice, and the system uses zero-shot voice conversion / cloning
(Seed-VC v2, EZ-VC, or OmniVoice) to generate ideal pronunciation examples in the
user's own voice timbre. Users then practice and compare via spectrograms,
waveforms, and pitch contours.

Reference audio is pre-generated once via generate_references.py and
shipped as static assets in reference_audio/.

Usage:
    python app.py [--seed-vc-path ./seed-vc] [--ez-vc-path ./EZ-VC] [--omnivoice-path ./OmniVoice]
                  [--compile] [--no-vc] [--no-share] [--port 7860]
"""

import argparse
import html
import os
import gradio as gr
import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(APP_DIR, ".env"))
except ImportError:
    pass

from sentences.data import get_sentences, get_reference_audio_path
from audio_analysis import (
    create_spectrogram,
    create_waveform,
    create_pitch_contour,
    create_comparison,
)

# ──────────────────────────────────────────────
# Globals
# ──────────────────────────────────────────────
NO_VC_MODE = False

vc_wrappers = {
    "Seed-VC v2": None,
    "EZ-VC": None,
    "OmniVoice": None,
}

MODEL_CFG_SETTINGS = {
    "Seed-VC v2": {
        "minimum": 0.0,
        "maximum": 1.0,
        "value": 0.7,
        "step": 0.05,
        "info": "Seed-VC v2 CFG rate (0.0–1.0). Default 0.7 balances intelligibility and voice similarity.",
    },
    "EZ-VC": {
        "minimum": 0.5,
        "maximum": 3.0,
        "value": 2.0,
        "step": 0.1,
        "info": "EZ-VC CFG strength (0.5–3.0). Higher sticks closer to your voice timbre.",
    },
    "OmniVoice": {
        "minimum": 0.5,
        "maximum": 4.0,
        "value": 2.0,
        "step": 0.1,
        "info": "OmniVoice guidance scale (0.5–4.0). Default 2.0.",
    },
}


def init_omnivoice(omnivoice_path=None):
    """Initialize OmniVoice wrapper (lazy-loaded on first use)."""
    try:
        from omnivoice_wrapper import get_wrapper
        vc_wrappers["OmniVoice"] = get_wrapper(omnivoice_path=omnivoice_path)
        print("[App] OmniVoice wrapper initialized")
    except Exception as e:
        print(f"[App] WARNING: Could not initialize OmniVoice wrapper: {e}")
        vc_wrappers["OmniVoice"] = None


def init_seed_vc(seed_vc_path=None, compile_model=False):
    """Initialize Seed-VC v2 wrapper (lazy-loaded on first use)."""
    try:
        from seed_vc_wrapper import get_wrapper
        vc_wrappers["Seed-VC v2"] = get_wrapper(
            seed_vc_path=seed_vc_path, compile_model=compile_model
        )
        print("[App] Seed-VC v2 wrapper initialized")
    except Exception as e:
        print(f"[App] WARNING: Could not initialize Seed-VC v2 wrapper: {e}")
        vc_wrappers["Seed-VC v2"] = None


def init_ez_vc(ez_vc_path=None):
    """Initialize EZ-VC wrapper (lazy-loaded on first use)."""
    try:
        from ez_vc_wrapper import get_wrapper
        vc_wrappers["EZ-VC"] = get_wrapper(ez_vc_path=ez_vc_path)
        print("[App] EZ-VC wrapper initialized")
    except Exception as e:
        print(f"[App] WARNING: Could not initialize EZ-VC wrapper: {e}")
        vc_wrappers["EZ-VC"] = None


def unload_inactive_models(active_model):
    """Keep only the selected VC backend resident in GPU memory."""
    for model_name, wrapper in vc_wrappers.items():
        if model_name == active_model or wrapper is None:
            continue
        is_loaded = False
        if hasattr(wrapper, "is_loaded"):
            is_loaded = wrapper.is_loaded()
        elif hasattr(wrapper, "_is_loaded"):
            is_loaded = bool(wrapper._is_loaded)
        if not is_loaded:
            continue
        if hasattr(wrapper, "unload"):
            try:
                wrapper.unload()
            except Exception as e:
                print(f"[App] WARNING: Could not unload {model_name}: {e}")


# ──────────────────────────────────────────────
# Core pipeline functions
# ──────────────────────────────────────────────
def enroll_voice(audio_path, gender, vc_model, cfg_rate, progress=gr.Progress()):
    """Process voice enrollment and bulk-generate all target sentences."""
    if not audio_path:
        return "❌ Please record or upload a voice sample first.", None

    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=None)
        if len(y) == 0:
            return "❌ Uploaded audio file is empty.", None

        duration = len(y) / sr
        if duration < 3.0:
            return f"⚠️ Recording is too short ({duration:.1f}s). Please record at least 3–5 seconds.", None
        if duration > 30.0:
            return f"⚠️ Recording is too long ({duration:.1f}s). Please keep it under 30 seconds.", None
        if float(np.max(np.abs(y))) < 1e-4:
            return "⚠️ Recording appears to be silent. Please check your microphone and try again.", None

        status = (
            f"✅ Voice sample validated!\n"
            f"📊 Duration: {duration:.1f}s | Sample rate: {sr}Hz | Gender: {gender.title()}\n\n"
        )

        if NO_VC_MODE:
            status += (
                "ℹ️ Running in --no-vc fallback mode. Voice conversion is bypassed; "
                "the Practice tab will play the pre-generated native reference audio."
            )
            return status, audio_path

        wrapper = vc_wrappers.get(vc_model)
        if wrapper is None:
            return f"❌ Selected VC model ({vc_model}) is not available. Please check server logs.", None

        unload_inactive_models(vc_model)

        try:
            if hasattr(wrapper, "load"):
                progress(0.0, desc=f"Loading {vc_model} model...")
                wrapper.load()
        except Exception as e:
            print(f"[App] Error loading {vc_model}: {e}")
            return f"❌ Could not load {vc_model}:\n{e}", None

        sentences_to_convert = []
        for lang in ["en-US", "es-ES"]:
            lang_sentences = get_sentences(lang)
            for i, s in enumerate(lang_sentences):
                ref_path = get_reference_audio_path(lang, gender, i)
                if ref_path and os.path.exists(ref_path):
                    sentences_to_convert.append((lang, i, ref_path, s["text"]))

        total = len(sentences_to_convert)
        if total == 0:
            status += "⚠️ No reference audio files found to convert."
            return status, audio_path

        status += f"🎙️ {vc_model} active. Generating {total} practice sentences in your voice...\n"
        print(f"[App] Starting bulk voice generation ({vc_model}) for {total} sentences...")

        converted_count = 0
        errors = []

        for idx, (lang, i, ref_path, text) in enumerate(sentences_to_convert):
            desc = f"Generating ({lang}) {idx + 1}/{total}: {text[:28]}..."
            progress(idx / total, desc=desc)
            try:
                if vc_model == "OmniVoice":
                    wrapper.generate_tts(
                        text=text,
                        reference_audio_path=audio_path,
                        language=lang,
                        cfg_rate=cfg_rate,
                        use_cache=True,
                    )
                else:
                    wrapper.convert_voice(
                        source_audio_path=ref_path,
                        reference_audio_path=audio_path,
                        diffusion_steps=25,
                        length_adjust=1.0,
                        inference_cfg_rate=cfg_rate,
                        use_cache=True,
                    )
                converted_count += 1
            except Exception as e:
                print(f"[App] Error generating sentence {i + 1} in {lang} with {vc_model}: {e}")
                errors.append(f"{lang} #{i + 1}: {e}")

        progress(1.0, desc="Complete!")

        if converted_count == 0 and errors:
            return f"❌ Voice generation failed for {vc_model}:\n{errors[0]}", None

        if errors:
            status += (
                f"⚠️ Generated {converted_count}/{total} sentences "
                f"({len(errors)} failed; first error: {errors[0]})."
            )
        else:
            extra = " Custom TTS is also ready in the Practice tab!" if vc_model == "OmniVoice" else ""
            status += (
                f"⚡ Bulk-generation complete! All {converted_count} sentences are ready "
                f"in the Practice tab.{extra}"
            )

        return status, audio_path
    except Exception as e:
        return f"❌ Error processing voice enrollment: {str(e)}", None


def update_practice_sentences(language, gender, enrolled_audio, vc_model, cfg_rate):
    """Update all 10 sentence cards in the Practice tab using fast cache lookups."""
    sentences = get_sentences(language)
    outputs = []
    wrapper = vc_wrappers.get(vc_model)

    for i in range(10):
        sentence_html_val = ""
        ref_path = None
        ideal_path = None
        visible = i < len(sentences)

        if visible:
            s = sentences[i]
            text = s["text"]
            focus = s["focus"]
            diff = "Beginner" if i < 5 else "Intermediate"
            escaped_text = html.escape(text)
            escaped_focus = html.escape(focus)
            sentence_html_val = f"""
            <div style="padding: 10px 15px; border-radius: 6px; background: #1a1a2e; border: 1px solid #162447;">
                <p style="color: #e94560; font-size: 0.8em; font-weight: bold; text-transform: uppercase; margin: 0 0 3px 0; letter-spacing: 0.5px;">Sentence #{i + 1} — {diff}</p>
                <h3 style="font-size: 1.3em; margin: 0 0 5px 0; color: #ffffff; line-height: 1.2;">{escaped_text}</h3>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="background: #0f3460; color: #e94560; font-size: 0.75em; padding: 2px 6px; border-radius: 4px; font-weight: bold;">Focus</span>
                    <p style="color: #c5c5c5; margin: 0; font-size: 0.85em;">{escaped_focus}</p>
                </div>
            </div>
            """
            ref_path = get_reference_audio_path(language, gender, i)

            if NO_VC_MODE:
                ideal_path = ref_path
            elif wrapper is not None and enrolled_audio:
                try:
                    if vc_model == "OmniVoice":
                        ideal_path = wrapper.get_cached_path(
                            text=text,
                            reference_audio_path=enrolled_audio,
                            language=language,
                            cfg_rate=cfg_rate,
                        )
                    elif ref_path is not None:
                        ideal_path = wrapper.get_cached_path(
                            source_audio_path=ref_path,
                            reference_audio_path=enrolled_audio,
                            diffusion_steps=25,
                            length_adjust=1.0,
                            inference_cfg_rate=cfg_rate,
                        )
                except Exception as e:
                    print(f"[App] Cache lookup error for sentence {i}: {e}")
                    ideal_path = None

        outputs.append(gr.update(value=sentence_html_val, visible=visible))
        outputs.append(gr.update(value=ref_path, visible=visible))
        outputs.append(gr.update(value=ideal_path, visible=visible))
        outputs.append(gr.update(value=None, visible=visible))  # Clear attempt recording
        outputs.append(gr.update(value=None, visible=visible))  # Clear comparison plot
        outputs.append(gr.update(value="", visible=visible))    # Clear comparison status

    return outputs


def generate_custom_tts(text, language, enrolled_audio, vc_model, cfg_rate):
    """Generate custom cloned TTS using OmniVoice."""
    if vc_model != "OmniVoice":
        return None, "⚠️ Switch VC Model to OmniVoice to use Custom Sentence Generation."

    if not text or not text.strip():
        return None, "⚠️ Please enter a sentence to synthesize."

    if NO_VC_MODE:
        return None, "ℹ️ Custom TTS is unavailable in --no-vc fallback mode."

    if not enrolled_audio:
        return None, "⚠️ Please record/upload and enroll your voice in the Voice Enrollment tab first."

    wrapper = vc_wrappers.get("OmniVoice")
    if wrapper is None:
        return None, "❌ OmniVoice wrapper is not available. Please check server logs."

    try:
        unload_inactive_models("OmniVoice")
        ideal_path = wrapper.generate_tts(
            text=text.strip(),
            reference_audio_path=enrolled_audio,
            language=language,
            cfg_rate=cfg_rate,
            use_cache=False,
        )
        return ideal_path, "✅ Custom cloned audio generated!"
    except Exception as e:
        print(f"[App] Error generating custom TTS: {e}")
        return None, f"❌ Error generating custom TTS: {e}"


def on_vc_model_change(model_name):
    """Update CFG slider bounds/defaults and toggle OmniVoice Custom TTS panel."""
    cfg = MODEL_CFG_SETTINGS.get(model_name, MODEL_CFG_SETTINGS["Seed-VC v2"])
    slider_update = gr.update(
        minimum=cfg["minimum"],
        maximum=cfg["maximum"],
        value=cfg["value"],
        step=cfg["step"],
        info=cfg["info"],
    )
    custom_group_update = gr.update(visible=(model_name == "OmniVoice"))
    return slider_update, custom_group_update


def run_comparison(ideal_audio, attempt_audio):
    """Generate comparison visualizations."""
    if not ideal_audio and not attempt_audio:
        return (
            create_comparison(None, None),
            "❌ Please enroll/generate ideal audio and record an attempt first.",
        )

    fig = create_comparison(ideal_audio, attempt_audio)

    notes = []
    if not ideal_audio:
        notes.append("⚠️ No ideal audio yet — enroll your voice in the Voice Enrollment tab first.")
    if not attempt_audio:
        notes.append("⚠️ No attempt recorded — record or upload yourself speaking the sentence.")

    if not notes:
        notes.append("✅ Comparison generated! Compare the spectrograms, waveforms, and pitch contours.")
        notes.append("💡 Tip: Similar spectrogram shapes and pitch contours indicate closer pronunciation and intonation.")

    return fig, "\n".join(notes)


# ──────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────
def build_ui():
    """Build the Gradio Blocks UI."""
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

    default_cfg = MODEL_CFG_SETTINGS["Seed-VC v2"]

    with gr.Blocks(
        title="Accent Trainer — AI Voice Cloning",
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
                <p class="subtitle">Perfect your pronunciation with AI voice cloning — Seed-VC v2 · EZ-VC · OmniVoice</p>
            </div>
        """)

        # ── State ──
        enrolled_audio_state = gr.State(None)

        with gr.Tabs():
            # ═══════════════════════════════════════════
            # TAB 1: Voice Enrollment
            # ═══════════════════════════════════════════
            with gr.Tab("🎤 Voice Enrollment", id="enrollment"):
                gr.Markdown("""
                ### Step 1: Enroll Your Voice
                Record or upload a **5–25 second** sample of your natural speech.  
                This will be used to clone your voice timbre for ideal pronunciation examples.
                
                **Suggested enrollment phrases:**
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
                            info="Selects the native TTS reference voice",
                        )
                        vc_model_select = gr.Radio(
                            choices=["Seed-VC v2", "EZ-VC", "OmniVoice"],
                            value="Seed-VC v2",
                            label="⚙️ VC / Cloning Model",
                            info="Choose the voice conversion / cloning backend",
                        )
                        cfg_rate_slider = gr.Slider(
                            minimum=default_cfg["minimum"],
                            maximum=default_cfg["maximum"],
                            value=default_cfg["value"],
                            step=default_cfg["step"],
                            label="🎚️ CFG / Guidance Rate",
                            info=default_cfg["info"],
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
                Choose your target language, listen to the native reference and your cloned "ideal" voice, record your attempt, and run the visual comparison!
                """)

                with gr.Row():
                    language_select = gr.Radio(
                        choices=[
                            ("English (for Spanish speakers)", "en-US"),
                            ("Spanish (for English speakers)", "es-ES"),
                        ],
                        value="en-US",
                        label="🌍 Target Language",
                    )

                # Custom TTS (Visible when OmniVoice is active)
                with gr.Group(visible=False) as custom_tts_group:
                    gr.Markdown("### 🗣️ Custom Sentence Generation (OmniVoice)")
                    with gr.Row():
                        with gr.Column(scale=3):
                            custom_tts_text = gr.Textbox(
                                label="Enter your own sentence to practice",
                                placeholder="Type any English or Spanish sentence here...",
                            )
                        with gr.Column(scale=1):
                            custom_tts_btn = gr.Button("🗣️ Generate TTS", variant="primary")
                    custom_tts_status = gr.Textbox(
                        label="Custom TTS Status", interactive=False, lines=1
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            custom_tts_audio = gr.Audio(
                                label="🎯 Custom Cloned Audio (Your Voice)",
                                type="filepath",
                                interactive=False,
                            )
                        with gr.Column(scale=1):
                            custom_attempt_audio = gr.Audio(
                                sources=["microphone", "upload"],
                                type="filepath",
                                label="🎤 Your Custom Attempt",
                            )
                    with gr.Accordion("📊 Custom Sentence Comparison Analysis", open=False):
                        with gr.Row():
                            with gr.Column(scale=1):
                                custom_compare_btn = gr.Button(
                                    "📊 Analyze & Compare", variant="secondary", size="sm"
                                )
                                custom_compare_status = gr.Textbox(
                                    label="Comparison Notes", interactive=False, lines=3
                                )
                            with gr.Column(scale=2):
                                custom_compare_plot = gr.Plot(
                                    label="Spectrogram / Pitch Comparison"
                                )
                    gr.HTML("<div style='margin-bottom: 25px;'></div>")

                # Curated 10 Practice Sentences
                with gr.Group(visible=True) as standard_practice_group:
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
                                        sources=["microphone", "upload"],
                                        type="filepath",
                                        label="🎤 Your Attempt",
                                    )
                                    attempt_recorders.append(attempt_player)

                            with gr.Accordion("📊 Comparison Analysis", open=False):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        comp_btn = gr.Button(
                                            "📊 Analyze & Compare",
                                            variant="secondary",
                                            size="sm",
                                        )
                                        compare_btns.append(comp_btn)
                                        comp_status = gr.Textbox(
                                            label="Comparison Notes",
                                            interactive=False,
                                            lines=3,
                                        )
                                        comparison_statuses.append(comp_status)
                                    with gr.Column(scale=2):
                                        comp_plot = gr.Plot(
                                            label="Spectrogram / Pitch Comparison"
                                        )
                                        comparison_plots.append(comp_plot)

                            gr.HTML("<div style='margin-bottom: 25px;'></div>")

            # ═══════════════════════════════════════════
            # TAB 3: Visualizations
            # ═══════════════════════════════════════════
            with gr.Tab("📊 Analysis", id="analysis"):
                gr.Markdown("""
                ### Detailed Audio Analysis
                Upload or record audio for standalone spectrogram, waveform, and pitch contour analysis.
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
                
                1. **🗣️ Voice Enrollment** — Record or upload a short sample of your natural speech.
                
                2. **🔊 Pre-generated References** — 10 sentences per language have been 
                   pre-recorded using Google Cloud TTS (Chirp 3 HD voices). These are static 
                   assets — no external TTS API calls are needed at runtime.
                
                3. **🎯 Voice Cloning Backends** — Choose between:
                   - **Seed-VC v2** — Zero-shot diffusion voice conversion preserving native pronunciation + your timbre
                   - **EZ-VC** — XEUS discrete speech units + F5-TTS flow matching voice conversion
                   - **OmniVoice** — Omnilingual zero-shot voice-cloning TTS (also unlocks custom text practice!)
                
                4. **📊 Comparison** — Record yourself attempting the sentence, then compare
                   mel spectrograms, waveforms, and pitch contours side by side.
                
                ---
                
                ### Languages
                
                - **English** (for Spanish speakers) — TH, SH, V/B, schwa, consonant clusters
                - **Spanish** (for English speakers) — rolled R, vowel purity, J sound, -ción endings
                
                10 sentences per language × 2 genders = 40 pre-generated reference audio files.
                """)

        # ──────────────────────────────────────────────
        # Event Listeners & Event Bindings
        # ──────────────────────────────────────────────
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

        practice_inputs = [
            language_select,
            gender_select,
            enrolled_audio_state,
            vc_model_select,
            cfg_rate_slider,
        ]

        # TAB 1: Voice Enrollment
        enroll_btn.click(
            fn=enroll_voice,
            inputs=[enrollment_audio, gender_select, vc_model_select, cfg_rate_slider],
            outputs=[enrollment_status, enrolled_audio_state],
        ).then(
            fn=update_practice_sentences,
            inputs=practice_inputs,
            outputs=practice_outputs,
        )

        gender_select.change(
            fn=update_practice_sentences,
            inputs=practice_inputs,
            outputs=practice_outputs,
        )

        vc_model_select.change(
            fn=on_vc_model_change,
            inputs=[vc_model_select],
            outputs=[cfg_rate_slider, custom_tts_group],
        ).then(
            fn=update_practice_sentences,
            inputs=practice_inputs,
            outputs=practice_outputs,
        )

        cfg_rate_slider.release(
            fn=update_practice_sentences,
            inputs=practice_inputs,
            outputs=practice_outputs,
        )

        # TAB 2: Practice
        language_select.change(
            fn=update_practice_sentences,
            inputs=practice_inputs,
            outputs=practice_outputs,
        )

        custom_tts_btn.click(
            fn=generate_custom_tts,
            inputs=[
                custom_tts_text,
                language_select,
                enrolled_audio_state,
                vc_model_select,
                cfg_rate_slider,
            ],
            outputs=[custom_tts_audio, custom_tts_status],
        )

        custom_compare_btn.click(
            fn=run_comparison,
            inputs=[custom_tts_audio, custom_attempt_audio],
            outputs=[custom_compare_plot, custom_compare_status],
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

        # Initial load
        demo.load(
            fn=update_practice_sentences,
            inputs=practice_inputs,
            outputs=practice_outputs,
        )

        # ── Footer ──
        gr.HTML("""
            <div style="text-align: center; padding: 20px; color: #666; font-size: 0.9em;">
                Accent Trainer Demo — Built with Seed-VC v2, EZ-VC, OmniVoice & Google Cloud TTS
            </div>
        """)

    return demo


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    global NO_VC_MODE

    parser = argparse.ArgumentParser(description="Accent Trainer Demo")
    parser.add_argument(
        "--seed-vc-path", type=str, default=None, help="Path to cloned seed-vc repo"
    )
    parser.add_argument(
        "--ez-vc-path", type=str, default=None, help="Path to cloned EZ-VC repo"
    )
    parser.add_argument(
        "--omnivoice-path", type=str, default=None, help="Path to cloned OmniVoice repo"
    )
    parser.add_argument(
        "--compile", action="store_true", help="Use torch.compile for faster Seed-VC inference"
    )
    parser.add_argument(
        "--no-vc",
        action="store_true",
        help="Skip loading VC models (reference-audio-only fallback mode for testing)",
    )
    parser.add_argument(
        "--no-share",
        action="store_true",
        help="Disable Gradio public share tunnel",
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port for Gradio server"
    )
    args = parser.parse_args()

    NO_VC_MODE = bool(args.no_vc)

    if not NO_VC_MODE:
        init_seed_vc(seed_vc_path=args.seed_vc_path, compile_model=args.compile)
        init_ez_vc(ez_vc_path=args.ez_vc_path)
        init_omnivoice(omnivoice_path=args.omnivoice_path)
    else:
        print("[App] Running in reference-audio-only mode (--no-vc flag set)")

    demo = build_ui()
    try:
        demo.queue(default_concurrency_limit=1, max_size=20)
    except TypeError:
        demo.queue(concurrency_count=1, max_size=20)

    demo.launch(
        share=not args.no_share,
        server_name="0.0.0.0",
        server_port=args.port,
        show_error=True,
        allowed_paths=[APP_DIR],
    )


if __name__ == "__main__":
    main()
