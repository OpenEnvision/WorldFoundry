"""WorldFoundry pipeline for resident in-tree DreamX-World inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from worldfoundry.core.io import write_video

from ..pipeline_utils import PipelineABC


class DreamXWorld5BCamPipeline(PipelineABC):
    """Image-conditioned camera interaction with a resident 5B core."""

    MODEL_ID = "dreamx-world-5b-cam"
    MODEL_PATH_OPTION = "checkpoint_source"

    def __init__(
        self,
        *,
        checkpoint_source: str | Path | None = None,
        wan_model_path: str | Path | None = None,
        device: str = "cuda",
        model_id: str = MODEL_ID,
        **options: Any,
    ) -> None:
        super().__init__(model_id=model_id, device=device, **options)
        self.checkpoint_source = checkpoint_source
        self.wan_model_path = wan_model_path
        self._realtime_session: Any = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: dict[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "DreamXWorld5BCamPipeline":
        options = cls._runtime_options(model_path, required_components, kwargs)
        resolved_model_id = cls._resolve_model_id(options, model_id=model_id)
        checkpoint_source = options.pop(
            "checkpoint_source",
            options.pop("checkpoint_dir", options.pop("model_path", None)),
        )
        wan_model_path = options.pop(
            "wan_model_path", options.pop("base_model_path", None)
        )
        cls._strip_framework_loading_options(options)
        return cls(
            checkpoint_source=checkpoint_source,
            wan_model_path=wan_model_path,
            device=device,
            model_id=resolved_model_id,
            **options,
        )

    def _ensure_realtime_session(self) -> Any:
        if self._realtime_session is None:
            from ...synthesis.visual_generation.dreamx_world.realtime import (
                DreamXWorldRealtimeSession,
            )

            self._realtime_session = DreamXWorldRealtimeSession(
                self.checkpoint_source,
                wan_model_path=self.wan_model_path,
            )
        return self._realtime_session

    def prepare_realtime(self) -> dict[str, Any]:
        session = self._ensure_realtime_session()
        return session.prepare()

    def configure_realtime(
        self,
        images: Any,
        prompt: str = "",
        seed: int = 42,
        fps: int = 16,
        num_frames: int = 33,
        height: int = 704,
        width: int = 1280,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        **_: Any,
    ) -> dict[str, Any]:
        if isinstance(images, (str, Path)):
            with Image.open(images) as source:
                images = source.convert("RGB")
        if not isinstance(images, Image.Image):
            raise ValueError("DreamX-World realtime requires a PIL image or image path.")
        return self._ensure_realtime_session().configure(
            images,
            prompt=prompt,
            seed=seed,
            fps=fps,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

    def stream_realtime(
        self,
        interactions: Sequence[str] | None = None,
        prompt: str | None = None,
        realtime_segments: Sequence[Mapping[str, Any]] | None = None,
        seed: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        return self._ensure_realtime_session().generate(
            interactions=list(interactions or ()),
            control_segments=realtime_segments,
            seed=seed,
            prompt=prompt,
        )

    def realtime_next_output_frames(self) -> int:
        return int(self._ensure_realtime_session().next_output_frames())

    def reset_realtime(self) -> None:
        if self._realtime_session is not None:
            self._realtime_session.reset()

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        return self(*args, **kwargs)

    def __call__(
        self,
        images: Any = None,
        prompt: str = "",
        interactions: Sequence[str] | None = None,
        fps: int = 16,
        num_frames: int = 33,
        height: int = 704,
        width: int = 1280,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        seed: int = 42,
        output_path: str | Path | None = None,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        self.configure_realtime(
            images=images,
            prompt=prompt,
            seed=seed,
            fps=fps,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs,
        )
        result = self.stream_realtime(interactions=interactions, seed=seed, **kwargs)
        if output_path is not None:
            frames = result.get("frames")
            if frames is None:
                raise RuntimeError("DreamX-World generation did not return video frames.")
            write_video(frames, output_path, fps=fps)
            result = dict(result)
            result.update(
                status="ok",
                artifact_kind="generated_video",
                artifact_path=str(output_path),
                fps=int(fps),
                num_output_frames=len(frames),
            )
        return result if return_dict else result["frames"]


class DreamXWorld5BARPipeline(DreamXWorld5BCamPipeline):
    """Distilled causal DreamX-World pipeline with persistent latent/KV state."""

    MODEL_ID = "dreamx-world-5b"

    def _ensure_realtime_session(self) -> Any:
        if self._realtime_session is None:
            from ...synthesis.visual_generation.dreamx_world.ar_realtime import (
                DreamXWorldARRealtimeSession,
            )

            self._realtime_session = DreamXWorldARRealtimeSession(
                self.checkpoint_source,
                wan_model_path=self.wan_model_path,
            )
        return self._realtime_session

    def configure_realtime(
        self,
        images: Any,
        prompt: str = "",
        seed: int = 42,
        fps: int = 16,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if isinstance(images, (str, Path)):
            with Image.open(images) as source:
                images = source.convert("RGB")
        if not isinstance(images, Image.Image):
            raise ValueError("DreamX-World AR requires a PIL image or image path.")
        return self._ensure_realtime_session().configure(
            images,
            prompt=prompt,
            seed=seed,
            fps=fps,
            **kwargs,
        )


__all__ = ["DreamXWorld5BARPipeline", "DreamXWorld5BCamPipeline"]
