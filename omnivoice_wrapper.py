"""OmniVoice wrapper for zero-shot voice-cloned text-to-speech."""

import gc
import hashlib
import importlib.util
import os
import sys
import threading
import time
import soundfile as sf
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass

OMNIVOICE_PATH = os.environ.get("OMNIVOICE_PATH", os.path.join(BASE_DIR, "OmniVoice"))
if not os.path.isabs(OMNIVOICE_PATH):
    OMNIVOICE_PATH = os.path.abspath(os.path.join(BASE_DIR, OMNIVOICE_PATH))

LOCAL_MODEL_PATH = os.environ.get("OMNIVOICE_MODEL_PATH", os.path.join(BASE_DIR, "OmniVoice_Model"))
if not os.path.isabs(LOCAL_MODEL_PATH):
    LOCAL_MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, LOCAL_MODEL_PATH))

CACHE_DIR = os.path.join(BASE_DIR, "cached_tts", "omnivoice")
os.makedirs(CACHE_DIR, exist_ok=True)

_LANG_MAP = {
    "en-US": "en",
    "es-ES": "es",
    "en": "en",
    "es": "es",
}


def _file_fingerprint(path):
    """Build a stable cache key component incorporating file path, mtime, and size."""
    abs_path = os.path.abspath(path)
    try:
        st = os.stat(abs_path)
        return f"{abs_path}:{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        return abs_path


class OmniVoiceWrapper:
    """Wrapper around k2-fsa/OmniVoice for zero-shot voice cloning TTS."""

    def __init__(self, omnivoice_path=None):
        resolved = omnivoice_path or OMNIVOICE_PATH
        if not os.path.isabs(resolved):
            resolved = os.path.abspath(os.path.join(BASE_DIR, resolved))
        self.omnivoice_path = resolved
        self.model = None
        self.cache_dir = CACHE_DIR
        self._is_loaded = False
        self._lock = threading.RLock()
        self._prompt_cache = {}

    def _ensure_path(self):
        if BASE_DIR not in sys.path:
            sys.path.insert(0, BASE_DIR)
        if os.path.exists(self.omnivoice_path) and self.omnivoice_path not in sys.path:
            sys.path.append(self.omnivoice_path)

    def load(self):
        with self._lock:
            if self._is_loaded:
                return

            self._ensure_path()

            # Enable hf_transfer only if the package is actually installed
            if importlib.util.find_spec("hf_transfer") is not None:
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            else:
                os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

            if os.environ.get("HF_ENDPOINT") == "https://hf-mirror.com":
                del os.environ["HF_ENDPOINT"]

            print("[OmniVoice] Loading model (this might take a while)...")
            from omnivoice import OmniVoice

            if torch.cuda.is_available():
                device_map = "cuda:0"
                dtype = torch.float16
            else:
                device_map = "cpu"
                dtype = torch.float32

            if os.path.exists(LOCAL_MODEL_PATH):
                print(f"[OmniVoice] Found local model at {LOCAL_MODEL_PATH}, loading from disk...")
                self.model = OmniVoice.from_pretrained(
                    LOCAL_MODEL_PATH,
                    device_map=device_map,
                    dtype=dtype,
                )
            else:
                from huggingface_hub import snapshot_download
                print("[OmniVoice] Ensuring model files are downloaded...")
                snapshot_download(repo_id="k2-fsa/OmniVoice", max_workers=2)
                self.model = OmniVoice.from_pretrained(
                    "k2-fsa/OmniVoice",
                    device_map=device_map,
                    dtype=dtype,
                )

            self._is_loaded = True
            print("[OmniVoice] Model loaded successfully.")

    def unload(self):
        with self._lock:
            if not self._is_loaded:
                return

            print("[OmniVoice] Unloading model to free VRAM...")
            self._prompt_cache.clear()
            if self.model is not None and hasattr(self.model, "_asr_pipe"):
                self.model._asr_pipe = None
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._is_loaded = False

    def is_loaded(self):
        return self._is_loaded

    def _get_voice_clone_prompt(self, reference_audio_path):
        """Build or fetch a cached VoiceClonePrompt so ASR + audio tokenization run once per voice."""
        fp = _file_fingerprint(reference_audio_path)
        prompt = self._prompt_cache.get(fp)
        if prompt is not None:
            return prompt

        prompt = self.model.create_voice_clone_prompt(ref_audio=reference_audio_path)
        self._prompt_cache[fp] = prompt
        return prompt

    def _compute_cache_path(self, text, reference_audio_path, language=None, cfg_rate=2.0):
        ref_fp = _file_fingerprint(reference_audio_path)
        lang_norm = _LANG_MAP.get(language, language) if language else ""
        cfg_val = round(min(max(float(cfg_rate), 0.5), 5.0), 4)
        digest = hashlib.md5(
            f"omnivoice:{text.strip()}:{ref_fp}:{lang_norm}:{cfg_val}".encode()
        ).hexdigest()
        return os.path.join(self.cache_dir, f"omnivoice_{digest}.wav")

    def get_cached_path(self, text, reference_audio_path, language=None, cfg_rate=2.0):
        """Return cached TTS output path if already computed, else None."""
        if not text or not reference_audio_path:
            return None
        path = self._compute_cache_path(
            text=text,
            reference_audio_path=reference_audio_path,
            language=language,
            cfg_rate=cfg_rate,
        )
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
        return None

    def generate_tts(self, text, reference_audio_path, language=None, cfg_rate=2.0, use_cache=True):
        """Generate TTS using OmniVoice with cached voice cloning prompt."""
        if not text or not text.strip():
            raise ValueError("Text must not be empty.")
        if not reference_audio_path:
            raise ValueError("Reference audio path is required.")

        cfg_val = round(min(max(float(cfg_rate), 0.5), 5.0), 4)
        lang_arg = _LANG_MAP.get(language, language) if language else None

        if use_cache:
            output_path = self._compute_cache_path(
                text=text,
                reference_audio_path=reference_audio_path,
                language=language,
                cfg_rate=cfg_val,
            )
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
        else:
            digest = hashlib.md5(f"{text}:{reference_audio_path}:{time.time_ns()}".encode()).hexdigest()[:12]
            output_path = os.path.join(self.cache_dir, f"omnivoice_custom_{digest}.wav")

        with self._lock:
            self.load()
            if not self.model:
                raise RuntimeError("OmniVoice model failed to load.")

            print(f"[OmniVoice] Generating TTS for: '{text[:60]}'")
            prompt = self._get_voice_clone_prompt(reference_audio_path)

            with torch.no_grad():
                audios = self.model.generate(
                    text=text.strip(),
                    language=lang_arg,
                    voice_clone_prompt=prompt,
                    guidance_scale=cfg_val,
                )

            sample_rate = getattr(self.model, "sampling_rate", 24000) or 24000
            sf.write(output_path, audios[0], sample_rate)
            return output_path


_wrapper = None


def get_wrapper(omnivoice_path=None):
    """Get or create the global OmniVoiceWrapper singleton."""
    global _wrapper
    if _wrapper is None:
        _wrapper = OmniVoiceWrapper(omnivoice_path=omnivoice_path)
    return _wrapper
