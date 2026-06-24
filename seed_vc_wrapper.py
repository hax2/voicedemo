"""Seed-VC v2 wrapper for programmatic voice conversion.

Wraps the Seed-VC v2 model (from the Plachtaa/seed-vc repo) for
programmatic inference. Loads the model once at startup and provides
a simple `convert_voice()` interface.

Expects the seed-vc repo to be cloned at SEED_VC_PATH (default: ./seed-vc).
"""

import os
import sys
import hashlib
import numpy as np
import torch
import torchaudio

# Path to the cloned seed-vc repository
SEED_VC_PATH = os.environ.get("SEED_VC_PATH", os.path.join(os.path.dirname(__file__), "seed-vc"))

# Cache directory for converted audio
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cached_vc")
os.makedirs(CACHE_DIR, exist_ok=True)


class SeedVCWrapper:
    """Wrapper around Seed-VC v2 for voice conversion."""

    def __init__(self, seed_vc_path=None, compile_model=False):
        """Initialize the Seed-VC v2 wrapper.

        Args:
            seed_vc_path: Path to the cloned seed-vc repo.
            compile_model: Whether to use torch.compile for faster inference.
        """
        self.seed_vc_path = seed_vc_path or SEED_VC_PATH
        self.vc_wrapper = None
        self.device = None
        self.dtype = torch.float16
        self.compile_model = compile_model
        self._loaded = False

    def _ensure_path(self):
        """Add seed-vc to sys.path if not already there."""
        if self.seed_vc_path not in sys.path:
            sys.path.insert(0, self.seed_vc_path)

    def load(self):
        """Load the Seed-VC v2 model. Call once at startup."""
        if self._loaded:
            return

        self._ensure_path()

        import yaml
        from hydra.utils import instantiate
        from omegaconf import DictConfig

        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"[SeedVC] Loading model on device: {self.device}")

        # Load V2 config
        config_path = os.path.join(self.seed_vc_path, "configs", "v2", "vc_wrapper.yaml")
        cfg = DictConfig(yaml.safe_load(open(config_path, "r")))

        # Instantiate wrapper
        self.vc_wrapper = instantiate(cfg)

        # Load checkpoints (auto-downloads from HuggingFace)
        # The default paths are handled internally by the wrapper
        self.vc_wrapper.load_checkpoints()
        self.vc_wrapper.to(self.device)
        self.vc_wrapper.eval()

        # Monkey-patch self.vc_wrapper.cfm.inference to remove unsupported sway_sampling/amo_sampling parameters
        if hasattr(self.vc_wrapper, 'cfm') and hasattr(self.vc_wrapper.cfm, 'inference'):
            original_inference = self.vc_wrapper.cfm.inference
            def patched_inference(*args, **kwargs):
                kwargs.pop('sway_sampling', None)
                kwargs.pop('amo_sampling', None)
                return original_inference(*args, **kwargs)
            self.vc_wrapper.cfm.inference = patched_inference
            print("[SeedVC] Patched cfm.inference to remove unsupported sway_sampling and amo_sampling kwargs")

        # Setup AR caches
        self.vc_wrapper.setup_ar_caches(
            max_batch_size=1,
            max_seq_len=4096,
            dtype=self.dtype,
            device=self.device
        )

        # Optional: compile for faster inference
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

    def convert_voice(
        self,
        source_audio_path,
        reference_audio_path,
        diffusion_steps=25,
        length_adjust=1.0,
        inference_cfg_rate=0.7,
        output_path=None,
        use_cache=True,
    ):
        """Convert voice from source audio to match reference speaker.

        This is the core function: takes TTS-generated reference audio
        (correct pronunciation in target language) and converts it to
        sound like the user's voice (from their enrollment sample).

        Args:
            source_audio_path: Path to the TTS reference audio (the "correct" pronunciation).
            reference_audio_path: Path to the user's voice sample (enrollment).
            diffusion_steps: Number of diffusion steps (25 default, 30-50 for best quality).
            length_adjust: Speed adjustment factor (1.0 = same speed).
            inference_cfg_rate: CFG rate for inference (0.7 default).
            output_path: Optional path for output file. Auto-generated if None.
            use_cache: Whether to use cached results.

        Returns:
            str: Path to the converted WAV file.
        """
        if not self._loaded:
            self.load()

        # Generate cache key
        cache_key = hashlib.md5(
            f"{source_audio_path}:{reference_audio_path}:{diffusion_steps}:{length_adjust}:{inference_cfg_rate}".encode()
        ).hexdigest()

        if output_path is None:
            output_path = os.path.join(CACHE_DIR, f"{cache_key}.wav")

        # Check cache
        if use_cache and os.path.exists(output_path):
            return output_path

        # The Seed-VC v2 cfm.py expects inference_cfg_rate to be an iterable (e.g., [0.5, 0.5])
        cfg_rate_arg = inference_cfg_rate
        if isinstance(cfg_rate_arg, (float, int)):
            cfg_rate_arg = [float(cfg_rate_arg), float(cfg_rate_arg)]

        # Run voice conversion using the wrapper's convert_voice method
        with torch.no_grad():
            converted_audio = self.vc_wrapper.convert_voice(
                source_audio_path=source_audio_path,
                target_audio_path=reference_audio_path,
                diffusion_steps=diffusion_steps,
                length_adjust=length_adjust,
                inference_cfg_rate=cfg_rate_arg,
                device=self.device,
                dtype=self.dtype,
            )
            output_sr = self.vc_wrapper.sr

        # Handle output format
        if isinstance(converted_audio, torch.Tensor):
            if converted_audio.dim() == 1:
                converted_audio = converted_audio.unsqueeze(0)
            torchaudio.save(output_path, converted_audio.cpu(), output_sr)
        elif isinstance(converted_audio, np.ndarray):
            import soundfile as sf
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
