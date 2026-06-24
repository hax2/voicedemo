import os
import hashlib
import torch
import soundfile as sf
import time

CACHE_DIR = "cached_tts/omnivoice"
os.makedirs(CACHE_DIR, exist_ok=True)

class OmniVoiceWrapper:
    def __init__(self):
        self.model = None
        self._is_loaded = False

    def load(self):
        if self._is_loaded:
            return
        
        # Use HuggingFace mirror to prevent hanging downloads
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        print("[OmniVoice] Loading model (this might take a while)...")
        from omnivoice import OmniVoice
        self.model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice", 
            device_map="cuda:0", 
            dtype=torch.float16
        )
        self._is_loaded = True
        print("[OmniVoice] Model loaded successfully.")

    def generate_tts(self, text, reference_audio_path, use_cache=True):
        """
        Generate TTS using OmniVoice with voice cloning.
        """
        self.load()
        
        # Build cache key
        if use_cache and reference_audio_path and os.path.exists(reference_audio_path):
            with open(reference_audio_path, "rb") as f:
                ref_hash = hashlib.md5(f.read()).hexdigest()
            text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
            cache_name = f"omnivoice_{ref_hash[:10]}_{text_hash[:10]}.wav"
            cache_path = os.path.join(CACHE_DIR, cache_name)
            
            if os.path.exists(cache_path):
                return cache_path
        else:
            cache_path = os.path.join(CACHE_DIR, f"omnivoice_temp_{int(time.time())}.wav")

        print(f"[OmniVoice] Generating TTS for: '{text}'")
        try:
            audio = self.model.generate(
                text=text,
                ref_audio=reference_audio_path
            )
            
            sf.write(cache_path, audio[0], 24000)
            return cache_path
        except Exception as e:
            print(f"[OmniVoice] Error generating TTS: {e}")
            raise

def get_wrapper():
    wrapper = OmniVoiceWrapper()
    wrapper.load()
    return wrapper
