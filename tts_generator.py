"""Google Cloud TTS reference audio generator.

Uses the Cloud Text-to-Speech REST API with Chirp 3 HD voices
(Algenib for male, Achernar for female) to generate high-quality
reference pronunciation audio.
"""

import base64
import hashlib
import json
import os
import requests
import soundfile as sf
import numpy as np
import io

from sentences.data import get_voice_name

# Cache directory for generated TTS audio
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cached_tts")
os.makedirs(CACHE_DIR, exist_ok=True)

# Google TTS API endpoint
TTS_API_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"


def _get_cache_path(text, voice_name):
    """Generate a deterministic cache file path for a text+voice combo."""
    key = f"{voice_name}:{text}"
    filename = hashlib.md5(key.encode()).hexdigest() + ".wav"
    return os.path.join(CACHE_DIR, filename)


def generate_reference_audio(
    text,
    language_code,
    gender="male",
    api_key=None,
):
    """Generate reference audio using Google Cloud TTS Chirp 3 HD.

    Args:
        text: The text to synthesize.
        language_code: Language code (e.g., 'en-US', 'es-ES').
        gender: 'male' or 'female' to select voice.
        api_key: Google Cloud API key. Falls back to env var.

    Returns:
        str: Path to the generated WAV file.
    """
    if api_key is None:
        api_key = os.environ.get("GOOGLE_TTS_API_KEY", "")

    voice_name = get_voice_name(gender, language_code)
    if not voice_name:
        raise ValueError(f"No voice configured for gender={gender}, lang={language_code}")

    # Check cache first
    cache_path = _get_cache_path(text, voice_name)
    if os.path.exists(cache_path):
        return cache_path

    # Build request
    request_body = {
        "input": {"text": text},
        "voice": {
            "languageCode": language_code,
            "name": voice_name,
        },
        "audioConfig": {
            "audioEncoding": "LINEAR16",
            "sampleRateHertz": 22050,
        },
    }

    url = f"{TTS_API_URL}?key={api_key}"
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(request_body),
        timeout=30,
    )

    if response.status_code != 200:
        error_detail = response.text
        raise RuntimeError(
            f"Google TTS API error ({response.status_code}): {error_detail}"
        )

    # Decode audio from response
    result = response.json()
    audio_bytes = base64.b64decode(result["audioContent"])

    # Save to cache as WAV
    with open(cache_path, "wb") as f:
        f.write(audio_bytes)

    return cache_path


def generate_all_references(language_code, gender="male", api_key=None):
    """Pre-generate reference audio for all sentences in a language.

    Args:
        language_code: Target language code.
        gender: Voice gender.
        api_key: Google Cloud API key.

    Returns:
        list[dict]: List of {text, focus, audio_path} for each sentence.
    """
    from sentences.data import get_sentences

    sentences = get_sentences(language_code)
    results = []

    for sentence in sentences:
        try:
            audio_path = generate_reference_audio(
                text=sentence["text"],
                language_code=language_code,
                gender=gender,
                api_key=api_key,
            )
            results.append({
                "text": sentence["text"],
                "focus": sentence["focus"],
                "audio_path": audio_path,
            })
        except Exception as e:
            print(f"Warning: Failed to generate TTS for '{sentence['text'][:40]}...': {e}")
            results.append({
                "text": sentence["text"],
                "focus": sentence["focus"],
                "audio_path": None,
            })

    return results
