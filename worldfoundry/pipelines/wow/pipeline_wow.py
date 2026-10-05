"""WoW visual generation pipeline module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from PIL import Image

from ...operators.wow_operator import WoWOperator
from ...synthesis.visual_generation.wow.wow_synthesis import WoWSynthesis
from ..pipeline_utils import PipelineABC


class WoWArgs:
    """Parameters shared by the native WoW Wan and Cosmos2 routes."""

    def __init__(
        self,
        gpu: int = 0,
        steps: int = 50,
        seed: int = 42,
        num_frames: int = 41,
        no_tiled: bool = False,
        enable_vram_management: bool = True,
        no_vram_management: bool = False,
        persistent_param_gb: int = 70,
        runtime_backend: str = "wan",
        custom_checkpoint: str = "WoW_video_dit.pt",
        output_fps: int = 15,
        model_size: str = "2B",
        resolution: str = "720",
        fps: int = 16,
        dit_path: str | None = None,
        num_conditional_frames: int = 1,
        guidance: float = 7.0,
        num_sampling_step: int = 35,
        num_gpus: int = 1,
        disable_guardrail: bool = True,
        offload_guardrail: bool = False,
        disable_prompt_refiner: bool = True,
        offload_prompt_refiner: bool = False,
        negative_prompt: str | None = None,
    ) -> None:
        self.gpu = gpu
        self.steps = steps
        self.seed = seed
        self.num_frames = num_frames
        self.no_tiled = no_tiled
        self.enable_vram_management = enable_vram_management
        self.no_vram_management = no_vram_management
        self.persistent_param_gb = persistent_param_gb
        self.runtime_backend = runtime_backend
        self.custom_checkpoint = custom_checkpoint
        self.output_fps = output_fps
        self.model_size = model_size
        self.resolution = resolution
        self.fps = fps
        self.dit_path = dit_path
        self.num_conditional_frames = num_conditional_frames
        self.guidance = guidance
        self.num_sampling_step = num_sampling_step
        self.num_gpus = num_gpus
        self.disable_guardrail = disable_guardrail
        self.offload_guardrail = offload_guardrail
        self.disable_prompt_refiner = disable_prompt_refiner
        self.offload_prompt_refiner = offload_prompt_refiner
        self.negative_prompt = negative_prompt

    def replace(self, **overrides: Any) -> "WoWArgs":
        values = {
            name: getattr(self, name)
            for name in (
                "gpu",
                "steps",
                "seed",
                "num_frames",
                "no_tiled",
                "enable_vram_management",
                "no_vram_management",
                "persistent_param_gb",
                "runtime_backend",
                "custom_checkpoint",
                "output_fps",
                "model_size",
                "resolution",
                "fps",
                "dit_path",
                "num_conditional_frames",
                "guidance",
                "num_sampling_step",
                "num_gpus",
                "disable_guardrail",
                "offload_guardrail",
                "disable_prompt_refiner",
                "offload_prompt_refiner",
                "negative_prompt",
            )
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        return WoWArgs(**values)


class WoWPipeline(PipelineABC):
    """Public WoW pipeline backed by canonical diffusion runtimes."""

    MODEL_ID = "wow"

    def __init__(
        self,
        operator: Optional[WoWOperator] = None,
        synthesis_model: Optional[WoWSynthesis] = None,
        synthesis_args: Optional[WoWArgs] = None,
        device: str = "cuda",
        model_id: str = MODEL_ID,
    ) -> None:
        self.model_id = model_id
        self.operator = operator or WoWOperator()
        self.synthesis_model = synthesis_model
        self.synthesis_args = synthesis_args or WoWArgs()
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_path: Optional[Union[str, Mapping[str, Any]]] = None,
        required_components: Optional[Dict[str, Any]] = None,
        synthesis_model_path: str = "X-Humanoid/WoW-1-Wan-14B-600k",
        synthesis_args: Optional[WoWArgs] = None,
        device: Optional[str] = None,
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "WoWPipeline":
        """Create a WoWPipeline from local checkpoint components."""

        component_options: dict[str, Any] = dict(required_components or {})
        if isinstance(model_path, Mapping):
            component_options.update(model_path)
            model_path = (
                component_options.pop("model_path", None)
                or component_options.pop("pretrained_model_path", None)
                or component_options.pop("checkpoint_folder", None)
                or component_options.pop("repo_root", None)
            )
        synthesis_model_path = str(
            model_path or component_options.pop("synthesis_model_path", None) or synthesis_model_path
        )
        synthesis_args = component_options.pop("synthesis_args", synthesis_args)
        component_options.update(kwargs)
        component_options = cls._strip_framework_loading_options(component_options)

        if synthesis_args is None:
            synthesis_args = _args_from_options(component_options)

        if device is None:
            device = f"cuda:{synthesis_args.gpu}"

        synthesis_model = WoWSynthesis.from_pretrained(
            pretrained_model_path=synthesis_model_path,
            synthesis_args=synthesis_args,
            device=device,
            **component_options,
        )

        return cls(
            operator=WoWOperator(),
            synthesis_model=synthesis_model,
            synthesis_args=synthesis_args,
            device=device,
            model_id=model_id or cls.MODEL_ID,
        )

    def process(
        self,
        input_path: Union[str, Path, Image.Image],
        text_prompt: str,
    ) -> Dict[str, Any]:
        """Process and normalize input arguments and conditions for inference."""

        processed_perception = self.operator.process_perception(input_path)
        self.operator.get_interaction(text_prompt)
        try:
            processed_interaction = self.operator.process_interaction()
        finally:
            self.operator.delete_last_interaction()

        return {
            "input_image": processed_perception.get("input_image"),
            "input_path": processed_perception.get("input_path"),
            "interaction": processed_interaction["processed_prompt"],
        }

    def __call__(
        self,
        input_path: Union[str, Path, Image.Image, None] = None,
        text_prompt: Optional[str] = None,
        prompt: Optional[str] = None,
        images: Any = None,
        image: Any = None,
        video: Any = None,
        ref_image_path: Any = None,
        args: Optional[WoWArgs] = None,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
        num_frames: Optional[int] = None,
        frames: Optional[int] = None,
        no_tiled: Optional[bool] = None,
        output_path: str | Path | None = None,
        return_dict: bool = False,
        fps: Optional[int] = None,
        output_fps: Optional[int] = None,
        runtime_backend: Optional[str] = None,
        model_size: Optional[str] = None,
        resolution: Optional[str] = None,
        dit_path: Optional[str] = None,
        num_conditional_frames: Optional[int] = None,
        guidance: Optional[float] = None,
        num_sampling_step: Optional[int] = None,
        num_gpus: Optional[int] = None,
        disable_guardrail: Optional[bool] = None,
        disable_prompt_refiner: Optional[bool] = None,
        negative_prompt: Optional[str] = None,
        operator_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute the complete pipeline generation flow.

        Supports both the legacy ``input_path``/``text_prompt`` surface and the
        standard WorldFoundry runner surface (``images``/``video``/``output_path``).
        """

        del operator_kwargs
        if self.synthesis_model is None:
            raise RuntimeError("WoW synthesis model is not loaded. Use from_pretrained() first.")

        args = (args or self.synthesis_args).replace(
            steps=steps,
            seed=seed,
            num_frames=num_frames if num_frames is not None else frames,
            no_tiled=no_tiled,
            output_fps=output_fps,
            runtime_backend=runtime_backend,
            model_size=model_size,
            resolution=resolution,
            fps=fps,
            dit_path=dit_path,
            num_conditional_frames=num_conditional_frames,
            guidance=guidance,
            num_sampling_step=num_sampling_step,
            num_gpus=num_gpus,
            disable_guardrail=disable_guardrail,
            disable_prompt_refiner=disable_prompt_refiner,
            negative_prompt=negative_prompt,
        )

        text_prompt = text_prompt or prompt
        if not text_prompt:
            raise ValueError("WoW requires a text_prompt or prompt.")

        media_input = _first_available(input_path, images, image, video, ref_image_path)
        if media_input is None:
            raise ValueError("WoW requires input_path, images, video, or ref_image_path.")

        processed_data = self.process(media_input, text_prompt)
        result = self.synthesis_model.predict(
            input_image=processed_data["input_image"],
            input_path=processed_data["input_path"],
            text_prompt=processed_data["interaction"],
            synthesis_args=args,
            output_path=output_path,
            fps=fps,
            return_dict=True,
            **kwargs,
        )

        if return_dict:
            result.setdefault("model_id", self.model_id)
            return result
        video_result = result.get("video")
        if video_result is not None:
            return video_result
        artifact_path = result.get("artifact_path")
        return artifact_path if artifact_path else result


def _args_from_options(options: Mapping[str, Any]) -> WoWArgs:
    arg_names = {
        "gpu",
        "steps",
        "seed",
        "num_frames",
        "no_tiled",
        "enable_vram_management",
        "no_vram_management",
        "persistent_param_gb",
        "runtime_backend",
        "custom_checkpoint",
        "output_fps",
        "model_size",
        "resolution",
        "fps",
        "dit_path",
        "num_conditional_frames",
        "guidance",
        "num_sampling_step",
        "num_gpus",
        "disable_guardrail",
        "offload_guardrail",
        "disable_prompt_refiner",
        "offload_prompt_refiner",
        "negative_prompt",
    }
    return WoWArgs(**{key: value for key, value in options.items() if key in arg_names})


def _first_available(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        return value
    return None
