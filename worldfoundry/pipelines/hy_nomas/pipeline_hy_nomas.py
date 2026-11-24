"""Hy-Nomas public generation and resident Studio adapter over native components."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from threading import RLock
from typing import Any, cast

from worldfoundry.core.execution.realtime.contracts import RealtimeSpec
from worldfoundry.pipelines.native_diffusion_video import NativeTextToVideoPipeline


class HyNomasPipeline(NativeTextToVideoPipeline):
    """Generate ordered text/camera segments with an isolated cache per rollout."""

    MODEL_ID = "hy-nomas"
    OWNER = "Hy-Nomas"
    ALLOW_MODEL_ID_OVERRIDE = True
    CHECKPOINT_ROLES = ("transformer", "vae")
    DEFAULT_HEIGHT = 720
    DEFAULT_WIDTH = 1280
    DEFAULT_NUM_FRAMES = 120
    DEFAULT_NUM_INFERENCE_STEPS = 8
    DEFAULT_GUIDANCE_SCALE = 1.0
    DEFAULT_FPS = 24
    DEFAULT_SCHEDULER_OPTIONS: Mapping[str, object] = {"shift": 5.0}
    REQUEST_INPUT_DEFAULTS: Mapping[str, object] = {
        "segments": None,
        "segment_latent_lengths": None,
        "camera_extrinsics": None,
        "cache_noise_sigma": 0.206897,
        "kv_cache_quantization": "none",
        "max_cache_segments": None,
        "fps": 24,
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._realtime_lock = RLock()
        self._realtime_state: Any = None
        self._realtime_options: dict[str, Any] = {}
        self._camera_pose = None

    @classmethod
    def _checkpoint_overrides(
        cls,
        model_path: str | Mapping[str, Any] | None,
        options: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        result = dict(super()._checkpoint_overrides(model_path, options) or {})
        text_path = options.get("text_encoder_path")
        if isinstance(text_path, (str, Path)):
            result["text-encoder"] = str(text_path)
        return result or None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> HyNomasPipeline:
        options = cls._options(model_path, required_components, kwargs)
        if options.pop("low_memory", False):
            options["variant"] = "hy-nomas-low-memory"
        return cast(
            HyNomasPipeline,
            super().from_pretrained(model_path, required_components, device, model_id=model_id, **options),
        )

    def __call__(self, prompt: str | list[str] = "", *args: Any, **kwargs: Any) -> Any:
        if isinstance(prompt, list):
            if "segments" in kwargs:
                raise ValueError("pass ordered prompts through prompt or segments, not both")
            kwargs["segments"] = prompt
            prompt = prompt[0] if prompt else ""
        if self.native_pipeline.model_id == "hy-nomas-low-memory":
            kwargs.setdefault("kv_cache_quantization", "int8")
        return super().__call__(prompt, *args, **kwargs)

    def prepare_realtime(self) -> dict[str, Any]:
        decoder = self.native_pipeline.runner.components.decoder.model
        ratio = decoder.temporal_compression_ratio
        window = decoder.config.temporal_window_latents
        stride = window - decoder.config.temporal_overlap_latents
        return {
            "realtime_spec": RealtimeSpec(
                fps=24,
                first_chunk_frames=ratio,
                steady_chunk_frames=stride * ratio,
                first_input_frames=window * ratio,
                steady_input_frames=stride * ratio,
            ).to_payload()
        }

    def configure_realtime(
        self,
        images: Any = None,
        prompt: str = "",
        *,
        seed: int = 42,
        height: int = 720,
        width: int = 1280,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if images is not None:
            raise ValueError("Hy-Nomas realtime starts from text, not a reference image")
        with self._realtime_lock:
            self.reset_realtime()
            self._realtime_options = {"prompt": prompt, "seed": seed, "height": height, "width": width}
            if self.native_pipeline.model_id == "hy-nomas-low-memory":
                self._realtime_options["kv_cache_quantization"] = "int8"
            spec = RealtimeSpec.from_payload(self.prepare_realtime()["realtime_spec"])
            request, _ = self._prepare_call(**self._realtime_options, num_frames=spec.first_input_frames)
            self._realtime_state = self.native_pipeline.runner.new_session(request)
        return self.prepare_realtime()

    def stream_realtime(
        self,
        interactions: list[str] | None = None,
        *,
        prompt: str | None = None,
        realtime_segments: list[Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        from worldfoundry.core.media.processing.postprocess import VideoChunk
        from worldfoundry.operators.hy_nomas_operator import realtime_camera

        with self._realtime_lock:
            if self._realtime_state is None:
                raise RuntimeError("configure_realtime must run before generating segments")
            spec = RealtimeSpec.from_payload(self.prepare_realtime()["realtime_spec"])
            frames = spec.first_input_frames if not self._realtime_state.frame_offset else spec.steady_input_frames
            assert frames is not None
            camera, pose = realtime_camera(
                interactions or [],
                frames,
                pose=self._camera_pose,
                segments=realtime_segments,
            )
            options = dict(self._realtime_options)
            if prompt is not None:
                options["prompt"] = prompt
            options.update(num_frames=frames, camera_extrinsics=camera)
            if self.native_pipeline.model_id == "hy-nomas-low-memory":
                options["kv_cache_quantization"] = "int8"
            request, _ = self._prepare_call(**options)
            result = self.native_pipeline.runner.continue_session(request, self._realtime_state)
            self._realtime_options["prompt"] = options["prompt"]
            self._camera_pose = pose
            if not result.sample.numel():
                return None
            return {"frames": VideoChunk(frames=result.sample, layout="bcthw", value_range="-1,1"), "fps": 24}

    def reset_realtime(self) -> None:
        with self._realtime_lock:
            self._realtime_state = None
            self._realtime_options = {}
            self._camera_pose = None
