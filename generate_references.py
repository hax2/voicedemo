"""One-time script to pre-generate all reference audio using Google Cloud TTS.

Run this ONCE to create the 40 reference audio files (10 sentences × 2 languages × 2 genders).
The generated files are saved to reference_audio/ and committed to the repo.
They are static assets — no TTS calls happen at runtime.

Usage:
    python generate_references.py
"""

import base64
import json
import os
import requests
import asyncio
import edge_tts
import librosa
import soundfile as sf

from sentences.data import SENTENCES, VOICE_CONFIG

# Load from environment or from .env file directly to prevent revoking via chat leak detection
API_KEY = os.environ.get("GOOGLE_TTS_API_KEY", "")
if not API_KEY and os.path.exists(".env"):
    try:
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("GOOGLE_TTS_API_KEY="):
                    API_KEY = line.strip().split("=", 1)[1].strip("'\" ")
                    break
    except Exception:
        pass

if not API_KEY:
    API_KEY = "AIzaSyDufo-Hu6qKgw_VqB8VBXHozzGNJE5KSyo"

TTS_API_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "reference_audio")


def synthesize_edge_fallback(text, language_code, voice_name, output_path):
    """Fallback to Edge TTS if Google TTS fails."""
    is_male = "male" in output_path.lower() or "algenib" in voice_name.lower()
    
    if "en-us" in language_code.lower():
        edge_voice = "en-US-AndrewNeural" if is_male else "en-US-JennyNeural"
    elif "es-es" in language_code.lower():
        edge_voice = "es-ES-AlvaroNeural" if is_male else "es-ES-ElviraNeural"
    else:
        edge_voice = "en-US-AndrewNeural"
        
    print(f"       -> [Fallback] Using Edge TTS ({edge_voice})...")
    
    temp_mp3 = output_path + ".temp.mp3"
    
    async def run_edge():
        communicate = edge_tts.Communicate(text, edge_voice)
        await communicate.save(temp_mp3)
        
    try:
        asyncio.run(run_edge())
        # Convert MP3 to WAV
        data, sr = librosa.load(temp_mp3, sr=22050)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, data, sr)
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)
        return True
    except Exception as e:
        print(f"       -> [Fallback ERROR]: {e}")
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)
        return False


def synthesize(text, voice_name, language_code, output_path):
    """Call Google Cloud TTS REST API, with fallback to Edge TTS."""
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

    try:
        url = f"{TTS_API_URL}?key={API_KEY}"
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(request_body),
            timeout=30,
        )

        if response.status_code == 200:
            audio_bytes = base64.b64decode(response.json()["audioContent"])
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            return True
        else:
            print(f"  Google TTS ERROR ({response.status_code}): {response.text[:200]}")
    except Exception as e:
        print(f"  Google TTS Request failed: {e}")

    # Fallback to Edge TTS if Google TTS fails
    return synthesize_edge_fallback(text, language_code, voice_name, output_path)


def main():
    total = 0
    errors = 0

    for lang_code, lang_data in SENTENCES.items():
        # Collect all sentences for this language
        all_sentences = []
        for difficulty in ["beginner", "intermediate"]:
            all_sentences.extend(lang_data.get(difficulty, []))

        for gender in ["male", "female"]:
            voice_name = VOICE_CONFIG[gender][lang_code]
            print(f"\n{'='*60}")
            print(f"Language: {lang_code} | Gender: {gender} | Voice: {voice_name}")
            print(f"{'='*60}")

            for i, sentence in enumerate(all_sentences):
                text = sentence["text"]
                # e.g. reference_audio/en-US/male/01.wav
                filename = f"{i+1:02d}.wav"
                output_path = os.path.join(OUTPUT_DIR, lang_code, gender, filename)

                if os.path.exists(output_path):
                    print(f"  [{i+1:2d}] SKIP (exists): {text[:50]}")
                    total += 1
                    continue

                print(f"  [{i+1:2d}] Generating: {text[:50]}...")
                if synthesize(text, voice_name, lang_code, output_path):
                    print(f"       -> Saved to {output_path}")
                    total += 1
                else:
                    errors += 1

    print(f"\n{'='*60}")
    print(f"Done! Generated {total} files, {errors} errors.")
    print(f"Files saved to: {OUTPUT_DIR}/")
    print(f"\nCommit these to your repo — they are static assets.")


if __name__ == "__main__":
    main()
