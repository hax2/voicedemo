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
        self.cache_dir = CACHE_DIR
        self._is_loaded = False

    def load(self):
        if self._is_loaded:
            return
        
        # Enable hf_transfer for fast Rust-based reliable downloads
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        if "HF_ENDPOINT" in os.environ:
            del os.environ["HF_ENDPOINT"]
            
        print("[OmniVoice] Loading model (this might take a while)...")
        from omnivoice import OmniVoice
        
        # If the user cloned the model locally to bypass network hangs, use the local folder
        local_model_path = "./OmniVoice_Model"
        if torch.cuda.is_available():
            device_map = "cuda:0"
            dtype = torch.float16
        else:
            device_map = "cpu"
            dtype = torch.float32

        if os.path.exists(local_model_path):
            print(f"[OmniVoice] Found local model at {local_model_path}, loading from disk...")
            self.model = OmniVoice.from_pretrained(
                local_model_path, 
                device_map=device_map, 
                dtype=dtype
            )
        else:
            from huggingface_hub import snapshot_download
            print("[OmniVoice] Pre-downloading model files sequentially to avoid hang...")
            snapshot_download(repo_id="k2-fsa/OmniVoice", max_workers=1)
            self.model = OmniVoice.from_pretrained(
                "k2-fsa/OmniVoice", 
                device_map=device_map, 
                dtype=dtype
            )
        self._is_loaded = True
        print("[OmniVoice] Model loaded successfully.")

    def unload(self):
        if not self._is_loaded:
            return

        print("[OmniVoice] Unloading model to free VRAM...")
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._is_loaded = False

    def generate_tts(self, text, reference_audio_path, use_cache=True):
        """
        Generate TTS using OmniVoice with voice cloning.
        """
        self.load()
        
        if not self.model:
            raise RuntimeError("OmniVoice model failed to load.")
            
        import torch

        # Generate cache key
        digest = hashlib.md5(f"omnivoice:{text}:{reference_audio_path}".encode()).hexdigest()
        cache_key = f"omnivoice_{digest}.wav"
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        if use_cache and os.path.exists(cache_path):
            return cache_path
            
        print(f"[OmniVoice] Generating TTS for: '{text}'")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with torch.no_grad():
                audio = self.model.generate(
                    text=text,
                    ref_audio=reference_audio_path
                )
            
            sf.write(cache_path, audio[0], 24000)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return cache_path
        except Exception as e:
            print(f"[OmniVoice] Error generating TTS: {e}")
            raise

def get_wrapper():
    wrapper = OmniVoiceWrapper()
    return wrapper
