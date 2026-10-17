"""WorldFoundry pipeline for LingBot-World-V2 causal-fast inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from ..pipeline_utils import PipelineABC


class LingBotWorldV2Pipeline(PipelineABC):
    """Batch image-and-camera-to-video pipeline for LingBot-World-V2."""

    MODEL_ID = "lingbot-world-v2"
    MODEL_PATH_OPTION = "checkpoint_source"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._realtime_core: Any = None
        self._realtime_session: Any = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: dict[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "LingBotWorldV2Pipeline":
        from ...synthesis.visual_generation.lingbot_world_v2 import LingBotWorldV2Synthesis

        options = cls._runtime_options(model_path, required_components, kwargs)
        resolved_model_id = cls._resolve_model_id(options, model_id=model_id)
        synthesis = LingBotWorldV2Synthesis.from_pretrained(
            {**options, "model_id": resolved_model_id},
            device=device,
        )
        return cls(
            model_id=resolved_model_id,
            synthesis_model=synthesis,
            memory_module=None,
            device=device,
        )

    @staticmethod
    def _camera_controls(
        interactions: Sequence[Any] | Mapping[str, Any] | None,
        *,
        frame_num: int,
        height: int,
        width: int,
    ) -> tuple[Any, Any]:
        if not interactions or isinstance(interactions, Mapping):
            return None, None
        commands = [str(item) for item in interactions]
        from ...operators.lingbot_world_operator import TrajectoryGenerator

        return TrajectoryGenerator().generate(commands, frame_num, height, width)

    def __call__(
        self,
        images: Any = None,
        prompt: str = "",
        interactions: Sequence[Any] | Mapping[str, Any] | None = None,
        action_path: str | Path | None = None,
        input_dir: str | Path | None = None,
        output_path: str | Path | None = None,
        return_dict: bool = False,
        frame_num: int | None = None,
        num_frames: int | None = None,
        resize_H: int = 480,
        resize_W: int = 832,
        **kwargs: Any,
    ) -> Any:
        if self.synthesis_model is None:
            raise RuntimeError("LingBotWorldV2Pipeline is not initialized.")
        action_path = action_path or input_dir
        if frame_num is None:
            runtime = getattr(self.synthesis_model, "runtime", None)
            frame_num = int(getattr(runtime, "defaults", {}).get("frame_num", 361))
        frame_num = int(num_frames) if num_frames is not None else frame_num
        if interactions and not isinstance(interactions, Mapping) and (resize_H, resize_W) != (480, 832):
            raise ValueError("Generated LingBot-World-V2 camera controls use the official 480x832 calibration.")
        c2ws, intrinsics = self._camera_controls(
            interactions if action_path is None else None,
            frame_num=frame_num,
            height=resize_H,
            width=resize_W,
        )
        result = self.synthesis_model.predict(
            prompt=prompt,
            images=images,
            interactions=interactions,
            action_path=action_path,
            c2ws=c2ws,
            intrinsics=intrinsics,
            output_path=output_path,
            frame_num=frame_num,
            **kwargs,
        )
        # Distributed inference intentionally returns no payload on nonzero
        # ranks.  Those ranks still execute every collective and must not try
        # to index the rank-zero result while Studio gathers status.
        if result is None:
            return None
        return result if return_dict else result["artifact_path"]

    def stream(
        self,
        images: Any = None,
        prompt: str = "",
        interactions: Sequence[Any] | Mapping[str, Any] | None = None,
        action_path: str | Path | None = None,
        input_dir: str | Path | None = None,
        output_path: str | Path | None = None,
        return_dict: bool = False,
        frame_num: int | None = None,
        num_frames: int | None = None,
        resize_H: int = 480,
        resize_W: int = 832,
        *,
        fps: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run one causal-fast batch, retaining structured artifact output when requested."""
        return self(
            images=images,
            prompt=prompt,
            interactions=interactions,
            action_path=action_path,
            input_dir=input_dir,
            output_path=output_path,
            return_dict=return_dict,
            frame_num=frame_num,
            num_frames=num_frames,
            resize_H=resize_H,
            resize_W=resize_W,
            fps=fps,
            **kwargs,
        )

    def _ensure_realtime_session(self) -> Any:
        """Load one distributed resident core per Studio worker process."""

        if self._realtime_session is not None:
            return self._realtime_session
        if self.synthesis_model is None:
            raise RuntimeError("LingBotWorldV2Pipeline is not initialized.")

        import os

        import torch
        import torch.distributed as dist

        from ...runtime.local_checkpoint_cache import stage_checkpoint_for_realtime
        from ...synthesis.visual_generation.lingbot_world_v2.inference import (
            LingBotWorldV2Inference,
        )
        from ...synthesis.visual_generation.lingbot_world_v2.realtime import (
            LingBotWorldV2RealtimeSession,
            LingBotWorldV2SmallRealtimeSession,
        )

        runtime = self.synthesis_model.runtime
        checkpoint = stage_checkpoint_for_realtime(
            runtime._resolve_checkpoint(required=True),
            required_paths=runtime.required_checkpoint_paths,
            distributed=dist,
        )
        defaults = dict(runtime.defaults)
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if runtime.num_heads % world_size:
            raise ValueError(
                f"LingBot-World-V2 realtime world size must divide {runtime.num_heads} attention heads; "
                f"got world_size={world_size}."
            )
        if torch.cuda.is_available():
            local_rank = torch.cuda.current_device()
        else:
            local_rank = int(os.getenv("LOCAL_RANK", "0") or "0")
        self._realtime_core = LingBotWorldV2Inference(
            checkpoint,
            device_id=local_rank,
            rank=rank,
            t5_fsdp=bool(defaults.get("t5_fsdp", False)) and world_size > 1,
            dit_fsdp=bool(defaults.get("dit_fsdp", False)) and world_size > 1,
            use_sp=world_size > 1,
            t5_cpu=bool(defaults.get("t5_cpu", False)),
            convert_model_dtype=bool(defaults.get("convert_model_dtype", False)),
            local_attn_size=int(defaults.get("local_attn_size", 18)),
            sink_size=int(defaults.get("sink_size", 6)),
            **runtime.inference_options(),
        )
        session_type = (
            LingBotWorldV2SmallRealtimeSession
            if runtime.model_id == "lingbot-world-v2-1.3b"
            else LingBotWorldV2RealtimeSession
        )
        self._realtime_session = session_type(self._realtime_core)
        return self._realtime_session

    def prepare_realtime(self) -> dict[str, Any]:
        session = self._ensure_realtime_session()
        return {"realtime_spec": session.realtime_spec().to_payload()}

    def configure_realtime(
        self,
        images: Any,
        prompt: str = "",
        seed: int = 42,
        fps: int = 16,
        **_: Any,
    ) -> dict[str, Any]:
        if isinstance(images, (str, Path)):
            images = Image.open(images).convert("RGB")
        if not isinstance(images, Image.Image):
            raise ValueError("LingBot-World-V2 realtime requires a PIL image or image path.")
        return self._ensure_realtime_session().configure(
            image=images,
            prompt=str(prompt or ""),
            seed=seed,
            fps=fps,
        )

    def stream_realtime(
        self,
        prompt: str | None = None,
        interactions: Sequence[str] | None = None,
        realtime_segments: Sequence[Mapping[str, Any]] | None = None,
        seed: int = 42,
        **_: Any,
    ) -> dict[str, Any] | None:
        session = self._ensure_realtime_session()
        if prompt is not None:
            session.update_prompt(prompt)
        return session.generate(
            interactions=list(interactions or ()),
            control_segments=realtime_segments,
            seed=seed,
        )

    def realtime_next_output_frames(self) -> int:
        return int(self._ensure_realtime_session().next_output_frames())

    def reset_realtime(self) -> None:
        if self._realtime_session is not None:
            self._realtime_session.reset()


class LingBotWorldV2SmallPipeline(LingBotWorldV2Pipeline):
    """Separately selectable 1.3B model with the shared causal pipeline contract."""

    MODEL_ID = "lingbot-world-v2-1.3b"


__all__ = ["LingBotWorldV2Pipeline", "LingBotWorldV2SmallPipeline"]
