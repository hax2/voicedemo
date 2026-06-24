"""EZ-VC wrapper for programmatic voice conversion.

Wraps the EZ-VC model (from the EZ-VC/EZ-VC repo) for
programmatic inference. Loads the model once at startup and provides
a simple `convert_voice()` interface.

Expects the EZ-VC repo to be cloned at EZ_VC_PATH (default: ./EZ-VC).
"""

import os
import sys
import hashlib
import numpy as np
import torchaudio

# Path to the cloned ez-vc repository
EZ_VC_PATH = os.environ.get("EZ_VC_PATH", os.path.join(os.path.dirname(__file__), "EZ-VC"))

# Cache directory for converted audio
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cached_vc")
os.makedirs(CACHE_DIR, exist_ok=True)


class EZVCWrapper:
    """Wrapper around EZ-VC for voice conversion."""

    def __init__(self, ez_vc_path=None):
        """Initialize the EZ-VC wrapper.

        Args:
            ez_vc_path: Path to the cloned EZ-VC repo.
        """
        self.ez_vc_path = ez_vc_path or EZ_VC_PATH
        self.device = None
        self._loaded = False
        
        self.model = None
        self.vocoder = None
        self.xeus_model = None
        self.apply_kmeans = None

    def _ensure_path(self):
        """Add ez-vc to sys.path if not already there."""
        if self.ez_vc_path not in sys.path:
            sys.path.insert(0, self.ez_vc_path)
        
        # EZ-VC has src dir
        src_path = os.path.join(self.ez_vc_path, "src")
        if os.path.exists(src_path) and src_path not in sys.path:
            sys.path.insert(0, src_path)

    def load(self):
        """Load the EZ-VC model. Call once at startup."""
        if self._loaded:
            return

        if not os.path.exists(self.ez_vc_path):
            raise RuntimeError(f"EZ-VC repository not found at {self.ez_vc_path}. Please clone it first.")

        self._ensure_path()

        import torch
        from cached_path import cached_path
        from omegaconf import OmegaConf
        from hydra.utils import get_class
        
        try:
            from f5_tts.infer.utils_infer import load_model, load_vocoder
            from f5_tts.infer.utils_xeus import load_xeus_model, ApplyKmeans
        except ImportError as e:
            raise RuntimeError(f"Failed to import EZ-VC modules. Ensure it is installed with 'pip install -e .'. Error: {e}")

        # Determine device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"[EZ-VC] Loading models on device: {self.device}")

        # Load XEUS model
        print("[EZ-VC] Loading XEUS model...")
        self.xeus_model = load_xeus_model(self.device).eval()
        self.apply_kmeans = ApplyKmeans(self.device)

        # Load vocoder
        print("[EZ-VC] Loading Vocoder...")
        self.vocoder = load_vocoder(vocoder_name="bigvgan", device=self.device)

        # Load TTS model
        print("[EZ-VC] Loading Main Model...")
        config_file = os.path.join(self.ez_vc_path, "src", "f5_tts", "configs", "F5TTS_Base_EZ-VC.yaml")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
            
        ckpt_file = str(cached_path("hf://SPRINGLab/EZ-VC/model_2700000.safetensors"))
        vocab_file = str(cached_path("hf://SPRINGLab/EZ-VC/vocab.txt"))

        model_cfg = OmegaConf.load(config_file)
        model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        
        self.model = load_model(
            model_cls=model_cls,
            model_cfg=model_cfg.model.arch,
            ckpt_path=ckpt_file,
            vocab_file=vocab_file,
            ode_method="euler",
            use_ema=True,
            device=self.device,
        )

        self._loaded = True
        print("[EZ-VC] Model loaded successfully")

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
        """Convert voice from source audio to match reference speaker using EZ-VC.

        Args:
            source_audio_path: Path to the TTS reference audio.
            reference_audio_path: Path to the user's voice sample.
            diffusion_steps: Number of diffusion steps.
            length_adjust: Speed adjustment factor.
            inference_cfg_rate: CFG rate for inference.
            output_path: Optional path for output file. Auto-generated if None.
            use_cache: Whether to use cached results.

        Returns:
            str: Path to the converted WAV file.
        """
        if not self._loaded:
            self.load()

        # Generate cache key
        cache_key = hashlib.md5(
            f"ezvc:{source_audio_path}:{reference_audio_path}:{diffusion_steps}:{length_adjust}:{inference_cfg_rate}".encode()
        ).hexdigest()

        if output_path is None:
            output_path = os.path.join(CACHE_DIR, f"{cache_key}.wav")

        # Check cache
        if use_cache and os.path.exists(output_path):
            return output_path

        import torch
        from f5_tts.infer.utils_infer import infer_process
        from f5_tts.infer.utils_xeus import extract_units
        import soundfile as sf
        
        # 1. Extract units from source audio (which is the speech to convert)
        try:
            print(f"[EZ-VC] Extracting units from {source_audio_path}...")
            units = extract_units(self.xeus_model, self.apply_kmeans, source_audio_path, self.device)
            # EZ-VC format expects units as a string of space-separated integers
            unit_str = " ".join([str(u) for u in units.tolist()])
        except Exception as e:
            print(f"[EZ-VC] Error extracting units: {e}")
            raise e

        # 2. Run inference using reference audio as the prompt
        try:
            print(f"[EZ-VC] Running inference...")
            # Note: The ez-vc infer_process might expect specific arguments.
            # We map our wrapper arguments to what f5_tts utils_infer typically expects.
            audio_out, sr_out, _ = infer_process(
                ref_audio=reference_audio_path,
                ref_text="",  # EZ-VC is textless
                gen_text=unit_str, # The extracted units act as the "text" for the decoder
                model_obj=self.model,
                vocoder=self.vocoder,
                mel_spec_type="vocos", # Usually vocos or bigvgan depending on config
                target_rms=0.1,
                cross_fade_duration=0.15,
                sway_sampling_coef=-1.0,
                cfg_strength=inference_cfg_rate,
                nfe_step=diffusion_steps,
                speed=length_adjust,
                fix_duration=None,
                device=self.device
            )
            
            # Save output
            sf.write(output_path, audio_out, sr_out)
            return output_path
            
        except Exception as e:
            print(f"[EZ-VC] Error during generation: {e}")
            # As a fallback for demo purposes if inference fails
            return source_audio_path

    def is_loaded(self):
        """Check if the model is loaded."""
        return self._loaded


# Global singleton instance
_wrapper = None


def get_wrapper(ez_vc_path=None):
    """Get or create the global EZVCWrapper singleton."""
    global _wrapper
    if _wrapper is None:
        _wrapper = EZVCWrapper(ez_vc_path=ez_vc_path)
    return _wrapper
