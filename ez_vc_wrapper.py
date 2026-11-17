"""EZ-VC wrapper for programmatic voice conversion.

Wraps the EZ-VC model (from the EZ-VC/EZ-VC repo) for
programmatic inference. Loads the model lazily and provides
a simple `convert_voice()` interface.

Expects the EZ-VC repo to be cloned at EZ_VC_PATH (default: ./EZ-VC).
"""

from contextlib import contextmanager
import gc
import hashlib
import os
import sys
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass

# Propagate HF_TOKEN for huggingface_hub / cached_path if configured
if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

EZ_VC_PATH = os.environ.get("EZ_VC_PATH", os.path.join(BASE_DIR, "EZ-VC"))
if not os.path.isabs(EZ_VC_PATH):
    EZ_VC_PATH = os.path.abspath(os.path.join(BASE_DIR, EZ_VC_PATH))

CACHE_DIR = os.path.join(BASE_DIR, "cached_vc")
os.makedirs(CACHE_DIR, exist_ok=True)

# F5-TTS / EZ-VC has a 22s total context window (ref_audio + generated_audio).
# Clipping reference audio to <=12s matches upstream preprocess_ref_audio_text.
MAX_REF_DURATION_SEC = 12.0


def _file_fingerprint(path):
    """Build a stable cache key component incorporating file path, mtime, and size."""
    abs_path = os.path.abspath(path)
    try:
        st = os.stat(abs_path)
        return f"{abs_path}:{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        return abs_path


def _patch_ezvc_bigvgan():
    """Patch EZ-VC's bundled BigVGAN for compatibility with modern huggingface_hub."""
    import torch
    from huggingface_hub import hf_hub_download
    from third_party.BigVGAN.bigvgan import BigVGAN, load_hparams_from_json

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


class EZVCWrapper:
    """Wrapper around EZ-VC for voice conversion."""

    def __init__(self, ez_vc_path=None):
        resolved = ez_vc_path or EZ_VC_PATH
        if not os.path.isabs(resolved):
            resolved = os.path.abspath(os.path.join(BASE_DIR, resolved))
        self.ez_vc_path = resolved
        self.device = None
        self.unit_device = None
        self._loaded = False
        self._lock = threading.RLock()

        self.model = None
        self.vocoder = None
        self.xeus_model = None
        self.apply_kmeans = None
        self._unit_cache = {}
        self._ref_audio_cache = {}

    @contextmanager
    def _in_infer_dir(self):
        """Temporarily switch to EZ-VC's infer directory so relative 'xeus/*' paths resolve."""
        infer_dir = os.path.join(self.ez_vc_path, "src", "f5_tts", "infer")
        prev_cwd = os.getcwd()
        try:
            if os.path.isdir(infer_dir):
                os.chdir(infer_dir)
            yield
        finally:
            os.chdir(prev_cwd)

    def _configure_cuda(self, torch):
        """Use conservative CUDA settings for long-running Gradio inference."""
        if not torch.cuda.is_available():
            return

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        try:
            warmup = torch.nn.Conv1d(1, 1, 3).cuda().eval()
            with torch.inference_mode():
                warmup(torch.zeros(1, 1, 16, device="cuda"))
            del warmup
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[EZ-VC] WARNING: CUDA/cuDNN warmup failed: {e}")

    def _load_unit_extractor(self, device):
        with self._in_infer_dir():
            from f5_tts.infer.utils_xeus import ApplyKmeans, load_xeus_model

            print(f"[EZ-VC] Loading XEUS unit extractor on device: {device}")
            self.xeus_model = load_xeus_model(device).eval()
            self.apply_kmeans = ApplyKmeans(device)
            self.unit_device = device

    def _ensure_path(self):
        """Append EZ-VC directories to sys.path without shadowing voicedemo modules."""
        if BASE_DIR not in sys.path:
            sys.path.insert(0, BASE_DIR)

        candidates = [
            self.ez_vc_path,
            os.path.join(self.ez_vc_path, "src"),
            os.path.join(self.ez_vc_path, "src", "third_party", "BigVGAN"),
        ]
        for p in candidates:
            if os.path.exists(p) and p not in sys.path:
                sys.path.append(p)

    @staticmethod
    def _wrap_hf_auth_error(exc):
        msg = str(exc)
        if "GatedRepoError" in type(exc).__name__ or "SPRINGLab/EZ-VC" in msg and ("401" in msg or "403" in msg or "restricted" in msg or "gated" in msg.lower()):
            return RuntimeError(
                "Access to HuggingFace model 'SPRINGLab/EZ-VC' is gated. "
                "Please accept the license at https://huggingface.co/SPRINGLab/EZ-VC "
                "and set HF_TOKEN in your .env file or environment (or run 'huggingface-cli login')."
            )
        return None

    def load(self):
        """Load the EZ-VC model."""
        with self._lock:
            if self._loaded:
                return

            if not os.path.exists(self.ez_vc_path):
                raise RuntimeError(
                    f"EZ-VC repository not found at {self.ez_vc_path}. "
                    "Run 'bash setup_server.sh' to clone it."
                )

            self._ensure_path()

            import torch
            from cached_path import cached_path
            from hydra.utils import get_class
            from omegaconf import OmegaConf

            try:
                from f5_tts.infer.utils_infer import load_model, load_vocoder
                with self._in_infer_dir():
                    import f5_tts.infer.utils_xeus  # noqa: F401
            except ImportError as e:
                raise RuntimeError(
                    f"Failed to import EZ-VC modules. Run 'bash setup_server.sh' first. Error: {e}"
                ) from e
            except Exception as e:
                wrapped = self._wrap_hf_auth_error(e)
                if wrapped is not None:
                    raise wrapped from e
                raise

            self._configure_cuda(torch)

            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

            print(f"[EZ-VC] Loading models on device: {self.device}")

            self._load_unit_extractor(self.device)

            try:
                _patch_ezvc_bigvgan()
            except Exception as e:
                print(f"[EZ-VC] Note: Could not apply BigVGAN monkey-patch: {e}")

            print("[EZ-VC] Loading Vocoder...")
            self.vocoder = load_vocoder(vocoder_name="bigvgan", device=self.device)

            print("[EZ-VC] Loading Main Model...")
            config_file = os.path.join(
                self.ez_vc_path, "src", "f5_tts", "configs", "F5TTS_Base_EZ-VC.yaml"
            )
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Config file not found: {config_file}")

            try:
                ckpt_file = str(cached_path("hf://SPRINGLab/EZ-VC/model_2700000.safetensors"))
                vocab_file = str(cached_path("hf://SPRINGLab/EZ-VC/vocab.txt"))
            except Exception as e:
                wrapped = self._wrap_hf_auth_error(e)
                if wrapped is not None:
                    raise wrapped from e
                raise

            model_cfg = OmegaConf.load(config_file)
            model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")

            self.model = load_model(
                model_cls=model_cls,
                model_cfg=model_cfg.model.arch,
                ckpt_path=ckpt_file,
                mel_spec_type="bigvgan",
                vocab_file=vocab_file,
                ode_method="euler",
                use_ema=True,
                device=self.device,
            )

            self._loaded = True
            print("[EZ-VC] Model loaded successfully")

    def unload(self):
        """Unload the model to free VRAM."""
        with self._lock:
            if not self._loaded:
                return
            print("[EZ-VC] Unloading model to free VRAM...")
            self.model = None
            self.vocoder = None
            self.xeus_model = None
            self.apply_kmeans = None
            self.unit_device = None
            self._unit_cache.clear()
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._loaded = False

    def _prepare_reference_audio(self, reference_audio_path):
        """Trim silence and cap reference audio to <=12s so F5-TTS's 22s window never overflows."""
        fp = _file_fingerprint(reference_audio_path)
        cached = self._ref_audio_cache.get(fp)
        if cached and os.path.exists(cached):
            return cached

        import librosa
        import soundfile as sf

        y, sr = librosa.load(reference_audio_path, sr=16000)
        y_trimmed, _ = librosa.effects.trim(y, top_db=35)
        if len(y_trimmed) >= int(1.0 * sr):
            y = y_trimmed

        max_samples = int(MAX_REF_DURATION_SEC * sr)
        if len(y) > max_samples:
            y = y[:max_samples]

        digest = hashlib.md5(f"ezvc_ref:{fp}".encode()).hexdigest()
        prepped_path = os.path.join(CACHE_DIR, f"ezvc_ref_{digest}.wav")
        sf.write(prepped_path, y, sr)
        self._ref_audio_cache[fp] = prepped_path
        return prepped_path

    def _extract_units_with_recovery(self, audio_path):
        fp = _file_fingerprint(audio_path)
        if fp in self._unit_cache:
            return self._unit_cache[fp]

        import torch
        with self._in_infer_dir():
            from f5_tts.infer.utils_xeus import extract_units

            try:
                with torch.inference_mode():
                    units = extract_units(audio_path, self.xeus_model, self.apply_kmeans, self.unit_device)
            except RuntimeError as e:
                message = str(e)
                if "CUDNN_STATUS_NOT_INITIALIZED" not in message or self.unit_device != "cuda":
                    raise

                print("[EZ-VC] CUDA cuDNN failed during unit extraction; retrying XEUS on CPU.")
                self.xeus_model = None
                self.apply_kmeans = None
                gc.collect()
                torch.cuda.empty_cache()
                self._load_unit_extractor("cpu")
                with torch.inference_mode():
                    units = extract_units(audio_path, self.xeus_model, self.apply_kmeans, self.unit_device)

        if not units or not units.strip():
            raise ValueError(f"Could not extract speech units from audio: {audio_path}")

        self._unit_cache[fp] = units
        return units

    def _compute_cache_path(
        self,
        source_audio_path,
        reference_audio_path,
        diffusion_steps=25,
        length_adjust=1.0,
        inference_cfg_rate=2.0,
    ):
        src_fp = _file_fingerprint(source_audio_path)
        ref_fp = _file_fingerprint(reference_audio_path)
        cfg_val = round(float(inference_cfg_rate), 4)
        cache_key = hashlib.md5(
            f"ezvc:{src_fp}:{ref_fp}:{diffusion_steps}:{length_adjust}:{cfg_val}".encode()
        ).hexdigest()
        return os.path.join(CACHE_DIR, f"{cache_key}.wav")

    def get_cached_path(
        self,
        source_audio_path,
        reference_audio_path,
        diffusion_steps=25,
        length_adjust=1.0,
        inference_cfg_rate=2.0,
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
        inference_cfg_rate=2.0,
        output_path=None,
        use_cache=True,
    ):
        """Convert voice from source audio to match reference speaker using EZ-VC."""
        if output_path is None:
            output_path = self._compute_cache_path(
                source_audio_path=source_audio_path,
                reference_audio_path=reference_audio_path,
                diffusion_steps=diffusion_steps,
                length_adjust=length_adjust,
                inference_cfg_rate=inference_cfg_rate,
            )

        if use_cache and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path

        with self._lock:
            if not self._loaded:
                self.load()

            import soundfile as sf
            import torch
            from f5_tts.infer.utils_infer import infer_process

            prepped_ref_path = self._prepare_reference_audio(reference_audio_path)

            source_units = self._extract_units_with_recovery(source_audio_path)
            ref_units = self._extract_units_with_recovery(prepped_ref_path)

            with torch.inference_mode():
                audio_out, sr_out, _ = infer_process(
                    ref_audio=prepped_ref_path,
                    ref_text=ref_units,
                    gen_text=source_units,
                    model_obj=self.model,
                    vocoder=self.vocoder,
                    mel_spec_type="bigvgan",
                    target_rms=0.1,
                    cross_fade_duration=0.15,
                    sway_sampling_coef=-1.0,
                    cfg_strength=float(inference_cfg_rate),
                    nfe_step=int(diffusion_steps),
                    speed=float(length_adjust),
                    fix_duration=None,
                    device=self.device,
                )

            if audio_out is None:
                raise RuntimeError("EZ-VC returned empty audio output.")

            sf.write(output_path, audio_out, sr_out)
            return output_path

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
