"""Preview-only MAGI-2 audio-video inference using shared model components.

The 114B MoE streams layers from CPU memory to one GPU. Text encoding precedes
preview denoising; video and audio decoders are materialized afterwards.
Image conditioning and the upstream refiner stage are not implemented and
are rejected before loading weights.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ..pipeline_utils import PipelineABC

# The preview DiT was trained at temporal stride 8. The Turbo decoder expands
# each latent interval to 4 frames, so preview-only video runs at 12.5 fps.
# The optional upstream refiner doubles temporal resolution before 25 fps export.
_VAE_STRIDE = (8, 16, 16)
_VIDEO_LATENT_CHANNELS = 48
_AUDIO_LATENT_CHANNELS = 64
_AUDIO_LATENT_FPS = 25.0
_AUDIO_SAMPLE_RATE = 44100
_DEFAULT_FPS = 12.5  # 10s preview -> 32 latent frames -> 125 decoded frames


def _floor_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, int(value) // multiple * multiple)


def _is_module(value: Any) -> bool:
    return isinstance(value, torch.nn.Module)


def _resolve_ffmpeg() -> str | None:
    """Locate ffmpeg robustly (PATH, interpreter bin, imageio-ffmpeg)."""

    found = shutil.which("ffmpeg")
    if found:
        return found
    candidate = Path(sys.executable).parent / "ffmpeg"
    if candidate.is_file():
        return str(candidate)
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class NativeMagi2Pipeline(PipelineABC):
    """Public MAGI-2-preview pipeline over the ported single-GPU components."""

    MODEL_ID = "magi2-preview"
    GENERATION_TYPE = "t2v"
    DEFAULT_SHORT_EDGE = 512
    DEFAULT_ASPECT_RATIO = "16:9"
    DEFAULT_DURATION_SECONDS = 10.0
    DEFAULT_NUM_INFERENCE_STEPS = 100
    DEFAULT_REFINER_STEPS = 5
    DEFAULT_FPS = _DEFAULT_FPS
    DEFAULT_SHIFT = 7.0
    DEFAULT_VIDEO_GUIDANCE = 5.0
    DEFAULT_AUDIO_GUIDANCE = 7.0

    def __init__(
        self,
        *,
        preview_transformer: Any = None,
        refiner_transformer: Any = None,
        video_vae: Any = None,
        turbo_decoder: Any = None,
        audio_vae: Any = None,
        text_encoder: Any = None,
        data_proxy: Any = None,
        python_executable: str | None = None,
        transformers_overlay: str | Path | None = None,
        device: str = "cuda",
        model_id: str | None = None,
    ) -> None:
        super().__init__(model_id=model_id or self.MODEL_ID, device=device)
        self.preview_transformer = preview_transformer
        self.refiner_transformer = refiner_transformer
        self.video_vae = video_vae
        self.turbo_decoder = turbo_decoder
        self.audio_vae = audio_vae
        self.text_encoder = text_encoder
        self.data_proxy = data_proxy
        self.python_executable = python_executable or sys.executable
        self.transformers_overlay = transformers_overlay

    # ------------------------------------------------------------------ #
    # Loading (sequential residency; components built lazily).
    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "NativeMagi2Pipeline":
        """Wire the pipeline from a MAGI-2-preview checkpoint root.

        ``model_path`` points at the HF ``sand-ai/MAGI-2-preview`` layout with
        subdirs ``preview/``, ``refiner/``, ``vae/``, ``turbo_vae/``,
        ``stable-audio-open-1.0/`` (audio), ``text_encoder/``. Component modules
        are NOT all resident at once — the pipeline loads each stage's weights
        on demand during :meth:`generate` (see the sequential-residency note in
        the module docstring). Here we only record the checkpoint root and any
        pre-injected components.
        """

        options: dict[str, Any] = dict(model_path) if isinstance(model_path, Mapping) else {}
        options.update(required_components or {})
        options.update(kwargs)
        source = options.get("checkpoint_path", options.get("model_path", model_path))
        resolved_model_id = str(options.get("model_id") or model_id or cls.MODEL_ID)

        pipe = cls(
            preview_transformer=options.get("preview_transformer"),
            refiner_transformer=options.get("refiner_transformer"),
            video_vae=options.get("video_vae"),
            turbo_decoder=options.get("turbo_decoder"),
            audio_vae=options.get("audio_vae"),
            text_encoder=options.get("text_encoder"),
            data_proxy=options.get("data_proxy"),
            python_executable=options.get("python_executable"),
            transformers_overlay=options.get("transformers_overlay"),
            device=device,
            model_id=resolved_model_id,
        )
        pipe._checkpoint_root = Path(str(source)).expanduser() if source is not None else None
        return pipe

    def _ensure_components(
        self,
        *,
        load_preview: bool = True,
        load_decoders: bool = True,
    ) -> None:
        """Lazily load real weights from the checkpoint root (sequential residency).

        The 228GB preview MoE is loaded to CPU RAM and run with per-layer GPU
        offload; the small VAEs/encoder are loaded resident on the device.
        """

        root = getattr(self, "_checkpoint_root", None)
        if root is None:
            return
        device = torch.device(self.device)
        from worldfoundry.base_models.diffusion_model.models.networks.magi2.config import (
            Magi2PreviewConfig,
        )
        from worldfoundry.base_models.diffusion_model.models.networks.magi2.loading import (
            load_magi2_audio_vae_weights,
            load_magi2_dit_weights,
            load_magi2_turbo_weights,
        )

        if load_preview and self.preview_transformer is None:
            from worldfoundry.base_models.diffusion_model.models.networks.magi2.preview_dit import (
                Transformer as PreviewTransformer,
            )

            with torch.device("meta"):
                dit = PreviewTransformer(Magi2PreviewConfig())
            load_magi2_dit_weights(dit, root / "preview", strict=True)
            # Adapters/rope resident on GPU; the 40 MoE layers stream per-layer.
            dit.pre_adapter.to(device)
            dit.post_adapter.to(device)
            dit.block.layer_offload_device = device
            self.preview_transformer = dit.eval()

        if load_decoders and self.turbo_decoder is None:
            import glob as _glob

            from worldfoundry.base_models.diffusion_model.models.autoencoders.magi2 import (
                Magi2TurboDecoder,
            )

            cfg_files = _glob.glob(str(root / "turbo_vae" / "*.json"))
            ckpt = next(iter(_glob.glob(str(root / "turbo_vae" / "*.ckpt"))), None)
            if cfg_files and ckpt:
                turbo_cfg = __import__("json").loads(Path(cfg_files[0]).read_text())
                turbo = (
                    Magi2TurboDecoder.from_config(turbo_cfg)
                    if hasattr(Magi2TurboDecoder, "from_config")
                    else Magi2TurboDecoder(**turbo_cfg)
                )
                load_magi2_turbo_weights(turbo, ckpt)
                self.turbo_decoder = turbo.to(device).eval()

        if load_decoders and self.audio_vae is None:
            import glob as _glob

            from worldfoundry.base_models.diffusion_model.models.autoencoders.magi2 import (
                create_model_from_config,
            )

            audio_root = root / "stable-audio-open-1.0"
            audio_sft = next(iter(_glob.glob(str(audio_root / "*.safetensors"))), None)
            if audio_sft is not None:
                config_path = audio_root / "model_config.json"
                if not config_path.is_file():
                    raise FileNotFoundError(
                        f"MAGI-2 audio VAE config is missing: {config_path}"
                    )
                av = create_model_from_config(json.loads(config_path.read_text()))
                load_magi2_audio_vae_weights(av, audio_sft, strict=False)
                self.audio_vae = av.to(device).eval()

        if load_preview and self.text_encoder is None:
            from worldfoundry.base_models.diffusion_model.models.encoders.magi2_qwen35 import (
                Magi2Qwen35SubprocessTextEncoder,
            )

            self.text_encoder = Magi2Qwen35SubprocessTextEncoder(
                str(root / "text_encoder"),
                device=self.device,
                skip_layer=2,
                python_executable=self.python_executable,
                transformers_overlay=self.transformers_overlay,
            )

    # ------------------------------------------------------------------ #
    # Geometry / plan.
    # ------------------------------------------------------------------ #
    def _plan(self, *, short_edge: int, aspect_ratio: str, duration_seconds: float) -> dict[str, int]:
        ratios = {"21:9": (21, 9), "16:9": (16, 9), "4:3": (4, 3), "1:1": (1, 1), "3:4": (3, 4), "9:16": (9, 16)}
        if aspect_ratio not in ratios:
            raise ValueError(f"Unsupported MAGI-2 aspect ratio: {aspect_ratio}")
        if isinstance(short_edge, bool) or not isinstance(short_edge, int) or short_edge < 16:
            raise ValueError("MAGI-2 short_edge must be an integer of at least 16 pixels")
        if not math.isfinite(duration_seconds) or duration_seconds <= 0:
            raise ValueError("MAGI-2 duration_seconds must be positive and finite")
        aw, ah = ratios[aspect_ratio]
        if aw >= ah:
            height, width = short_edge, int(round(short_edge * aw / ah))
        else:
            width, height = short_edge, int(round(short_edge * ah / aw))
        unit = _VAE_STRIDE[1]  # spatial 16
        # Match the released 896x512 preview preset for the default 16:9 shape.
        height, width = _floor_to_multiple(height, unit), _floor_to_multiple(width, unit)
        frames = round(duration_seconds * self.DEFAULT_FPS * 2)
        video_latent_t = (frames - 1) // _VAE_STRIDE[0] + 1
        audio_latent_t = round(duration_seconds * _AUDIO_LATENT_FPS)
        return {
            "height": height,
            "width": width,
            "frames": (video_latent_t - 1) * 4 + 1,
            "video_latent_t": video_latent_t,
            "latent_h": height // _VAE_STRIDE[1],
            "latent_w": width // _VAE_STRIDE[2],
            "audio_latent_t": audio_latent_t,
        }

    # ------------------------------------------------------------------ #
    # Core generation (preview stage).
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str | None = None,
        short_edge: int = DEFAULT_SHORT_EDGE,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        duration_seconds: float = DEFAULT_DURATION_SECONDS,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        seed: int = 0,
        use_refiner: bool = False,
        return_latents: bool = False,
    ) -> dict[str, Any]:
        """Run the preview denoise and decode.

        Returns a dict with ``plan``, decoded ``video``/``audio`` tensors (when
        the VAEs are loaded), or raw ``video_latent``/``audio_latent`` when
        ``return_latents`` is set. Uses the injected components; callers that
        did not inject them must have set a checkpoint root via
        :meth:`from_pretrained` (weight loading wired in the integration seam).
        """

        if use_refiner:
            raise NotImplementedError("MAGI-2 refiner inference is not implemented; use preview-only T2V.")
        # Upstream explicitly supports only ten-second clips. Short latent
        # windows are not a valid way to reduce the released model's workload.
        if duration_seconds != self.DEFAULT_DURATION_SECONDS:
            raise ValueError("MAGI-2-preview supports only duration_seconds=10.0")
        if isinstance(num_inference_steps, bool) or not isinstance(num_inference_steps, int) or num_inference_steps < 1:
            raise ValueError("MAGI-2 num_inference_steps must be a positive integer")
        plan = self._plan(short_edge=short_edge, aspect_ratio=aspect_ratio, duration_seconds=duration_seconds)

        from worldfoundry.base_models.diffusion_model.models.networks.magi2 import (
            Magi2PreviewSampler,
            Magi2SamplingConfig,
        )
        from worldfoundry.base_models.diffusion_model.models.networks.magi2.sampler import (
            DEFAULT_NEGATIVE_PROMPT,
        )
        from worldfoundry.base_models.diffusion_model.schedulers.magi2 import (
            FlowUniPCMultistepScheduler,
        )

        # Denoising is the peak-memory stage. Keep the video/audio decoders on
        # CPU until the preview DiT has finished streaming all of its layers.
        for decoder in (self.turbo_decoder, self.video_vae, self.audio_vae):
            if _is_module(decoder):
                decoder.to("cpu")
        self._ensure_components(load_decoders=False)
        if self.preview_transformer is None:
            raise RuntimeError("preview_transformer is not initialized")
        device = torch.device(self.device)
        if self.data_proxy is None:
            from worldfoundry.base_models.diffusion_model.models.networks.magi2.data_proxy import (
                Magi2PreviewDataProxy,
                Magi2PreviewDataProxyConfig,
            )

            self.data_proxy = Magi2PreviewDataProxy(Magi2PreviewDataProxyConfig())
        cfg = Magi2SamplingConfig()

        # 1. Text encode (context = hidden_states[-3], 5120-d).
        context = self._encode_prompt(prompt)
        if negative_prompt is None:
            negative_prompt = DEFAULT_NEGATIVE_PROMPT
        null_context = self._encode_prompt(negative_prompt) if negative_prompt else torch.zeros_like(context[:, :0])
        if _is_module(self.preview_transformer):
            # A previous call offloaded these after denoising.
            self.preview_transformer.pre_adapter.to(device)
            self.preview_transformer.post_adapter.to(device)

        # 2. Match the official CUDA RNG draws for video then audio while
        # keeping this pipeline's seed local to the call.
        gen = torch.Generator(device=device).manual_seed(int(seed))
        video_latent = torch.randn(
            1, _VIDEO_LATENT_CHANNELS, plan["video_latent_t"], plan["latent_h"], plan["latent_w"],
            generator=gen, dtype=torch.float32, device=device,
        )
        audio_latent = torch.randn(
            1, plan["audio_latent_t"], _AUDIO_LATENT_CHANNELS,
            generator=gen, dtype=torch.float32, device=device,
        )

        # 3. Preview denoise via the sampler driving the DiT through the data proxy.
        video_scheduler = FlowUniPCMultistepScheduler()
        audio_scheduler = FlowUniPCMultistepScheduler()
        video_scheduler.set_timesteps(num_inference_steps, device=device, shift=cfg.shift)
        audio_scheduler.set_timesteps(num_inference_steps, device=device, shift=cfg.shift)
        video_t_list = list(video_scheduler.timesteps)

        model_forward = self._make_model_forward(self.preview_transformer, context, device)
        sampler = Magi2PreviewSampler(model_forward=model_forward, device=device, dtype=torch.bfloat16)
        cfg_config = cfg.to_cfg_config()
        video_latent, audio_latent = sampler.sample(
            video_t_list=video_t_list,
            latent=video_latent,
            audio_latent=audio_latent,
            txt_feat=context,
            null_txt_feat=null_context,
            video_scheduler=video_scheduler,
            audio_scheduler=audio_scheduler,
            cfg_config=cfg_config,
            progress=False,
        )

        # Every streamed MoE layer is back on CPU after ``sample``. Move the
        # small resident adapters as well, then materialize only the decoders.
        if _is_module(self.preview_transformer):
            self.preview_transformer.pre_adapter.to("cpu")
            self.preview_transformer.post_adapter.to("cpu")
        torch.cuda.empty_cache()

        result: dict[str, Any] = {"plan": plan}
        if return_latents:
            result["video_latent"] = video_latent
            result["audio_latent"] = audio_latent
            return result

        self._ensure_components(load_preview=False, load_decoders=True)
        for decoder in (self.turbo_decoder, self.video_vae, self.audio_vae):
            if _is_module(decoder):
                decoder.to(device)
        if _is_module(self.turbo_decoder):
            result["video"] = self.turbo_decoder.decode(video_latent)
        elif _is_module(self.video_vae) and hasattr(self.video_vae, "decode"):
            result["video"] = self.video_vae.decode(video_latent)
        if _is_module(self.audio_vae):
            from scipy.signal import resample

            # audio latent [1, L, 64] -> [1, 64, L] for the VAE decode contract.
            audio = self.audio_vae.decode(audio_latent.transpose(1, 2))
            # The DiT emits 25 tokens/s, expanded by 2048 to 51.2 kHz.
            # Match the official decoder's resampling before 44.1 kHz export.
            samples = int(audio.shape[-1] * 441 / 512)
            result["audio"] = torch.from_numpy(
                resample(audio.float().cpu().numpy(), samples, axis=-1)
            ).to(audio.device)
        return result

    def _make_model_forward(self, transformer: Any, context: torch.Tensor, device: torch.device):
        """Adapt the DiT + data proxy to the sampler's model_forward contract.

        Integration seam: the data proxy packs the cat([cond, uncond]) latents
        into the DiT's flat token stream and unpacks the velocity output back to
        (video, audio) latents. Wired concretely once the data proxy lands.
        """

        proxy = self.data_proxy

        def model_forward(*, x_t, audio_x_t, t, per_token_video_t, per_token_audio_t, context=None, **_):
            if proxy is None:
                raise NotImplementedError(
                    "model_forward requires the MAGI-2 data proxy to pack latents into the DiT token stream"
                )
            # The sampler bundles conditioning into a dict (`txt_feat` already
            # cat([cond, uncond]) plus ref features); the data proxy wants the
            # text tensor as `context` with ref fields as separate kwargs.
            ctx = context or {}
            txt_feat = ctx.get("txt_feat")
            inputs = proxy.build_model_inputs(
                latent=x_t,
                audio_latent=audio_x_t,
                context=txt_feat,
                per_token_video_t=per_token_video_t,
                per_token_audio_t=per_token_audio_t,
                t=t,
                audio_feat_len=ctx.get("audio_feat_len"),
                txt_feat_len=ctx.get("txt_feat_len"),
                ref_image_feat=ctx.get("ref_image_feat"),
                ref_image_feat_len=ctx.get("ref_image_feat_len"),
                ref_image_special_token_embedding=ctx.get("ref_image_special_token_embedding"),
            )
            dit_out = transformer(**inputs)
            return proxy.unpack_output(dit_out)

        return model_forward

    def _run_refiner(self, video_latent, context, null_context, plan, cfg, device):
        raise NotImplementedError("refiner stage wired once the refiner DiT + data proxy land")

    # ------------------------------------------------------------------ #
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        if self.text_encoder is None:
            raise RuntimeError("text_encoder is not initialized")
        return self.text_encoder.encode(prompt)

    def __call__(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        short_edge: int = DEFAULT_SHORT_EDGE,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        duration_seconds: float = DEFAULT_DURATION_SECONDS,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        seed: int = 0,
        use_refiner: bool = False,
        output_path: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if any(kwargs.get(key) is not None for key in ("images", "image", "image_path", "video", "video_path")):
            raise NotImplementedError("MAGI-2 currently supports text-to-video only; visual conditioning is not implemented.")
        result = self.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            short_edge=short_edge,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
            num_inference_steps=num_inference_steps,
            seed=seed,
            use_refiner=use_refiner,
        )
        if output_path is not None and "video" in result:
            result["output_path"] = self._write_mp4_with_audio(
                video=result["video"], audio=result.get("audio"),
                output_path=Path(output_path), fps=self.DEFAULT_FPS,
            )
        return result

    @staticmethod
    def _write_mp4_with_audio(*, video: torch.Tensor, audio: torch.Tensor | None, output_path: Path, fps: float) -> str:
        from worldfoundry.core.media.codecs.video import save_image_or_video_tensor

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if audio is None:
            save_image_or_video_tensor(video, str(output_path), fps=fps)
            return str(output_path)
        ffmpeg = _resolve_ffmpeg()
        with tempfile.TemporaryDirectory() as tmp:
            silent = Path(tmp) / "video.mp4"
            save_image_or_video_tensor(video, str(silent), fps=fps)
            wav = Path(tmp) / "audio.wav"
            _write_wav(audio, wav, sample_rate=_AUDIO_SAMPLE_RATE)
            if ffmpeg is None:
                save_image_or_video_tensor(video, str(output_path), fps=fps)
                shutil.copy(str(wav), str(output_path.with_suffix(".wav")))
                return str(output_path)
            subprocess.run(
                [
                    ffmpeg, "-y", "-i", str(silent), "-i", str(wav),
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                    "-af", "apad", "-t", str(video.shape[-3] / fps), str(output_path),
                ],
                check=True,
                capture_output=True,
            )
        return str(output_path)


def _write_wav(audio: torch.Tensor, path: Path, *, sample_rate: int) -> None:
    """Write a ``[channels, samples]`` / ``[1, C, L]`` / ``[C, 1, L]`` waveform to WAV."""

    import wave

    waveform = audio.detach().to("cpu", dtype=torch.float32)
    while waveform.ndim > 2:
        squeezed = False
        for dim in range(waveform.ndim):
            if waveform.shape[dim] == 1:
                waveform = waveform.squeeze(dim)
                squeezed = True
                break
        if not squeezed:
            waveform = waveform.reshape(waveform.shape[0], -1)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    channels, _ = waveform.shape
    pcm = (torch.clamp(waveform, -1.0, 1.0) * 32767.0).to(torch.int16).transpose(0, 1).contiguous().numpy()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(int(channels))
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())


__all__ = ["NativeMagi2Pipeline"]
