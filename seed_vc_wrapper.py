"""Seed-VC v2 wrapper for programmatic voice conversion.

Wraps the Seed-VC v2 model (from the Plachtaa/seed-vc repo) for
programmatic inference. Loads the model lazily and provides
a simple `convert_voice()` interface.

Expects the seed-vc repo to be cloned at SEED_VC_PATH (default: ./seed-vc).
"""

import gc
import hashlib
import os
import sys
import threading
import numpy as np
import soundfile as sf
import torch
import torchaudio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass

MIN_SAFE_TORCH_LOAD_VERSION = (2, 6)

# Path to the cloned seed-vc repository
SEED_VC_PATH = os.environ.get("SEED_VC_PATH", os.path.join(BASE_DIR, "seed-vc"))
if not os.path.isabs(SEED_VC_PATH):
    SEED_VC_PATH = os.path.abspath(os.path.join(BASE_DIR, SEED_VC_PATH))

# Cache directory for converted audio
CACHE_DIR = os.path.join(BASE_DIR, "cached_vc")
os.makedirs(CACHE_DIR, exist_ok=True)


def _file_fingerprint(path):
    """Build a stable cache key component incorporating file path, mtime, and size."""
    abs_path = os.path.abspath(path)
    try:
        st = os.stat(abs_path)
        return f"{abs_path}:{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        return abs_path


def _normalize_seedvc_cfg(inference_cfg_rate):
    """Clamp Seed-VC v2 CFG rate into its valid [0.0, 1.0] range."""
    if isinstance(inference_cfg_rate, (list, tuple)) and len(inference_cfg_rate) >= 2:
        intel = min(max(float(inference_cfg_rate[0]), 0.0), 1.0)
        sim = min(max(float(inference_cfg_rate[1]), 0.0), 1.0)
        return round(intel, 4), round(sim, 4)
    val = round(min(max(float(inference_cfg_rate), 0.0), 1.0), 4)
    return val, val


def _torch_version_tuple(version):
    core_version = version.split("+", 1)[0]
    parts = core_version.split(".")[:2]
    try:
        return tuple(int(part) for part in parts)
    except ValueError:
        return (0, 0)


def _require_safe_torch_load_version():
    if _torch_version_tuple(torch.__version__) >= MIN_SAFE_TORCH_LOAD_VERSION:
        return

    min_version = ".".join(str(part) for part in MIN_SAFE_TORCH_LOAD_VERSION)
    raise RuntimeError(
        "Seed-VC checkpoint loading requires PyTorch "
        f"{min_version}.0 or newer because older torch.load versions are blocked "
        "by upstream security checks for CVE-2025-32434. "
        f"Current torch version: {torch.__version__}. "
        'Run "bash setup_server.sh" to install the pinned torch/torchaudio versions.'
    )


def _patch_seedvc_bigvgan():
    """Patch Seed-VC's bundled BigVGAN for compatibility with modern huggingface_hub."""
    from huggingface_hub import hf_hub_download
    from modules.bigvgan.bigvgan import BigVGAN, load_hparams_from_json

    if getattr(BigVGAN, "_patched_from_pretrained", False):
        return

    @classmethod
    def patched_from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str = None,
        cache_dir: str = None,
        force_download: bool = False,
        proxies=None,
        resume_download: bool = False,
        local_files_only: bool = False,
        token=None,
        map_location: str = "cpu",
        strict: bool = False,
        use_cuda_kernel: bool = False,
        **model_kwargs,
    ):
        if os.path.isdir(model_id):
            config_file = os.path.join(model_id, "config.json")
            model_file = os.path.join(model_id, "bigvgan_generator.pt")
        else:
            dl_kwargs = dict(
                repo_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                token=token,
                local_files_only=local_files_only,
            )
            config_file = hf_hub_download(filename="config.json", **dl_kwargs)
            model_file = hf_hub_download(filename="bigvgan_generator.pt", **dl_kwargs)

        h = load_hparams_from_json(config_file)
        model = cls(h, use_cuda_kernel=use_cuda_kernel)
        checkpoint_dict = torch.load(model_file, map_location=map_location)
        try:
            model.load_state_dict(checkpoint_dict["generator"])
        except RuntimeError:
            model.remove_weight_norm()
            model.load_state_dict(checkpoint_dict["generator"])
        return model

    BigVGAN._from_pretrained = patched_from_pretrained
    BigVGAN._patched_from_pretrained = True


class SeedVCWrapper:
    """Wrapper around Seed-VC v2 for voice conversion."""

    def __init__(self, seed_vc_path=None, compile_model=False):
        resolved = seed_vc_path or SEED_VC_PATH
        if not os.path.isabs(resolved):
            resolved = os.path.abspath(os.path.join(BASE_DIR, resolved))
        self.seed_vc_path = resolved
        self.vc_wrapper = None
        self.device = None
        self.dtype = torch.float16
        self.compile_model = compile_model
        self._loaded = False
        self._lock = threading.RLock()

    def _ensure_path(self):
        """Append seed-vc to sys.path so it never shadows voicedemo modules."""
        if BASE_DIR not in sys.path:
            sys.path.insert(0, BASE_DIR)
        if self.seed_vc_path not in sys.path:
            sys.path.append(self.seed_vc_path)

    def load(self):
        """Load the Seed-VC v2 model."""
        with self._lock:
            if self._loaded:
                return

            if not os.path.exists(self.seed_vc_path):
                raise RuntimeError(
                    f"Seed-VC repository not found at {self.seed_vc_path}. "
                    "Run 'bash setup_server.sh' to clone it."
                )

            self._ensure_path()

            import yaml
            from hydra.utils import instantiate
            from omegaconf import DictConfig

            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.dtype = torch.float16
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self.dtype = torch.float16
            else:
                self.device = torch.device("cpu")
                self.dtype = torch.float32

            print(f"[SeedVC] Loading model on device: {self.device}")

            config_path = os.path.join(self.seed_vc_path, "configs", "v2", "vc_wrapper.yaml")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Seed-VC v2 config not found at {config_path}")

            with open(config_path, "r", encoding="utf-8") as f:
                cfg = DictConfig(yaml.safe_load(f))

            _require_safe_torch_load_version()
            _patch_seedvc_bigvgan()

            prev_cwd = os.getcwd()
            try:
                os.chdir(BASE_DIR)
                self.vc_wrapper = instantiate(cfg)
                self.vc_wrapper.load_checkpoints()
            finally:
                os.chdir(prev_cwd)

            self.vc_wrapper.to(self.device)
            self.vc_wrapper.eval()

            # Ensure cfm.inference tolerates legacy sway_sampling/amo_sampling kwargs
            if hasattr(self.vc_wrapper, "cfm") and hasattr(self.vc_wrapper.cfm, "inference"):
                original_inference = self.vc_wrapper.cfm.inference

                def patched_inference(*args, **kwargs):
                    kwargs.pop("sway_sampling", None)
                    kwargs.pop("amo_sampling", None)
                    return original_inference(*args, **kwargs)

                self.vc_wrapper.cfm.inference = patched_inference

            self.vc_wrapper.setup_ar_caches(
                max_batch_size=1,
                max_seq_len=4096,
                dtype=self.dtype,
                device=self.device,
            )

            if self.compile_model:
                try:
                    torch._inductor.config.coordinate_descent_tuning = True
                    torch._inductor.config.triton.unique_kernel_names = True
                    if hasattr(torch._inductor.config, "fx_graph_cache"):
                        torch._inductor.config.fx_graph_cache = True
                    self.vc_wrapper.compile_ar()
                    print("[SeedVC] Model compiled successfully")
                except Exception as e:
                    print(f"[SeedVC] Compilation failed (non-fatal): {e}")

            self._loaded = True
            print("[SeedVC] Model loaded successfully")

    def unload(self):
        """Unload the model to free VRAM."""
        with self._lock:
            if not self._loaded:
                return
            print("[SeedVC] Unloading model to free VRAM...")
            self.vc_wrapper = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._loaded = False

    def _compute_cache_path(
        self,
        source_audio_path,
        reference_audio_path,
        diffusion_steps=25,
        length_adjust=1.0,
        inference_cfg_rate=0.7,
        convert_style=False,
    ):
        intel_cfg, sim_cfg = _normalize_seedvc_cfg(inference_cfg_rate)
        src_fp = _file_fingerprint(source_audio_path)
        ref_fp = _file_fingerprint(reference_audio_path)
        cache_key = hashlib.md5(
            f"seedvc_v2:{src_fp}:{ref_fp}:{diffusion_steps}:{length_adjust}:{intel_cfg}:{sim_cfg}:{convert_style}".encode()
        ).hexdigest()
        return os.path.join(CACHE_DIR, f"{cache_key}.wav")

    def get_cached_path(
        self,
        source_audio_path,
        reference_audio_path,
        diffusion_steps=25,
        length_adjust=1.0,
        inference_cfg_rate=0.7,
        convert_style=False,
    ):
        """Return cached conversion output path if already computed, else None."""
        if not source_audio_path or not reference_audio_path:
            return None
        path = self._compute_cache_path(
            source_audio_path=source_audio_path,
            reference_audio_path=reference_audio_path,
            diffusion_steps=diffusion_steps,
            length_adjust=length_adjust,
            inference_cfg_rate=inference_cfg_rate,
            convert_style=convert_style,
        )
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
        return None

    def convert_voice(
        self,
        source_audio_path,
        reference_audio_path,
        diffusion_steps=25,
        length_adjust=1.0,
        inference_cfg_rate=0.7,
        output_path=None,
        use_cache=True,
        convert_style=False,
    ):
        """Convert voice timbre from source audio to match reference speaker.

        Uses Seed-VC v2's `convert_voice_with_streaming(..., convert_style=False)`
        by default so the native speaker's pronunciation/accent is preserved while
        cloning the enrolled speaker's timbre.
        """
        intel_cfg, sim_cfg = _normalize_seedvc_cfg(inference_cfg_rate)

        if output_path is None:
            output_path = self._compute_cache_path(
                source_audio_path=source_audio_path,
                reference_audio_path=reference_audio_path,
                diffusion_steps=diffusion_steps,
                length_adjust=length_adjust,
                inference_cfg_rate=(intel_cfg, sim_cfg),
                convert_style=convert_style,
            )

        if use_cache and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path

        with self._lock:
            if not self._loaded:
                self.load()

            if hasattr(self.vc_wrapper, "convert_voice_with_streaming"):
                generator = self.vc_wrapper.convert_voice_with_streaming(
                    source_audio_path=source_audio_path,
                    target_audio_path=reference_audio_path,
                    diffusion_steps=int(diffusion_steps),
                    length_adjust=float(length_adjust),
                    intelligebility_cfg_rate=intel_cfg,
                    similarity_cfg_rate=sim_cfg,
                    convert_style=convert_style,
                    anonymization_only=False,
                    device=self.device,
                    dtype=self.dtype,
                    stream_output=True,
                )
                full_audio = None
                for _, full_audio in generator:
                    pass
                if full_audio is None:
                    raise RuntimeError("Seed-VC v2 returned empty audio output.")
                output_sr, wav_array = full_audio
                sf.write(output_path, wav_array, output_sr)
                return output_path

            # Fallback for custom/older wrappers without convert_voice_with_streaming
            with torch.no_grad():
                converted_audio = self.vc_wrapper.convert_voice(
                    source_audio_path=source_audio_path,
                    target_audio_path=reference_audio_path,
                    diffusion_steps=int(diffusion_steps),
                    length_adjust=float(length_adjust),
                    inference_cfg_rate=[intel_cfg, sim_cfg],
                    device=self.device,
                    dtype=self.dtype,
                )
                output_sr = self.vc_wrapper.sr

            if isinstance(converted_audio, torch.Tensor):
                if converted_audio.dim() == 1:
                    converted_audio = converted_audio.unsqueeze(0)
                torchaudio.save(output_path, converted_audio.cpu(), output_sr)
            elif isinstance(converted_audio, np.ndarray):
                if converted_audio.ndim == 1:
                    sf.write(output_path, converted_audio, output_sr)
                else:
                    sf.write(output_path, converted_audio.T, output_sr)

            return output_path

    def is_loaded(self):
        """Check if the model is loaded."""
        return self._loaded


# Global singleton instance
_wrapper = None


def get_wrapper(seed_vc_path=None, compile_model=False):
    """Get or create the global SeedVCWrapper singleton."""
    global _wrapper
    if _wrapper is None:
        _wrapper = SeedVCWrapper(seed_vc_path=seed_vc_path, compile_model=compile_model)
    return _wrapper
