"""Base visual generation pipeline module."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional

from ...synthesis.visual_generation.memory.video import VideoArtifactMemory
from ...operators.runtime_video_operator import RuntimeVideoOperator
from ..pipeline_utils import PipelineABC


_CHECKPOINT_KEYS = ("ckpt_path", "checkpoint_path", "pretrained_model_path", "model_path")
_RUNTIME_KWARGS = {
    "config",
    "ckpt_path",
    "height",
    "width",
    "generation_type",
    "frames",
    "fps",
    "n_samples",
    "ddim_steps",
    "ddim_eta",
    "unconditional_guidance_scale",
    "seed",
    "device",
    "model_name",
}

_CALL_RUNTIME_ALIASES = {
    "num_frames": "frames",
    "frame_num": "frames",
    "num_inference_steps": "ddim_steps",
    "steps": "ddim_steps",
    "sample_steps": "ddim_steps",
    "infer_steps": "ddim_steps",
    "sampling_steps": "ddim_steps",
    "guidance_scale": "unconditional_guidance_scale",
    "cfg_scale": "unconditional_guidance_scale",
}


def _videocrafter_runtime_inputs(
    model_path: Any,
    required_components: Optional[Mapping[str, Any]],
    device: str,
    overrides: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Videocrafter runtime inputs helper function."""
    runtime_overrides = {
        key: value
        for source in (required_components or {}, overrides)
        for key, value in dict(source).items()
        if key in _RUNTIME_KWARGS
    }
    runtime_overrides.setdefault("device", device)

    checkpoint = model_path
    if isinstance(model_path, Mapping):
        values = dict(model_path)
        checkpoint = next((values[key] for key in _CHECKPOINT_KEYS if key in values and values[key]), None)
        runtime_overrides.update({key: value for key, value in values.items() if key in _RUNTIME_KWARGS})

    return checkpoint, runtime_overrides


class VideoCrafterPipelineBase(PipelineABC):
    """Shared WorldFoundry pipeline contract for VideoCrafter variants."""

    SYNTHESIS_CLS = None
    MEMORY_MODEL_ID = ""

    def __init__(
        self,
        operator: Optional[RuntimeVideoOperator] = None,
        synthesis_model=None,
        memory_module: Optional[VideoArtifactMemory] = None,
        device: str = "cuda",
    ) -> None:
        """Initialize the pipeline and configure runtime components."""
        if self.SYNTHESIS_CLS is None:
            raise TypeError("VideoCrafterPipelineBase subclasses must define SYNTHESIS_CLS.")
        self.synthesis_model = synthesis_model
        self.generation_type = getattr(
            synthesis_model,
            "generation_type",
            getattr(self.SYNTHESIS_CLS, "GENERATION_TYPE", "t2v"),
        )
        self.model_name = getattr(synthesis_model, "model_name", getattr(self.SYNTHESIS_CLS, "MODEL_NAME", None))
        self.operator = operator or RuntimeVideoOperator(generation_type=self.generation_type)
        self.memory_module = memory_module or VideoArtifactMemory(model_id=self.MEMORY_MODEL_ID)
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        lazy: bool = True,
        **kwargs: Any,
    ):
        """Load the pipeline from pretrained checkpoints and configurations."""
        checkpoint, generator_overrides = _videocrafter_runtime_inputs(
            model_path,
            required_components,
            device,
            kwargs,
        )
        synthesis_model = cls.SYNTHESIS_CLS.from_pretrained(
            pretrained_model_path=checkpoint,
            device=device,
            lazy=lazy,
            generator_overrides=generator_overrides,
        )
        return cls(
            operator=RuntimeVideoOperator(generation_type=synthesis_model.generation_type),
            synthesis_model=synthesis_model,
            memory_module=VideoArtifactMemory(model_id=cls.MEMORY_MODEL_ID),
            device=device,
        )

    def process(self, prompt: str = "", images=None, **kwargs) -> Dict[str, Any]:
        """Process and normalize input arguments and conditions for inference."""
        del kwargs
        self.operator.get_interaction(prompt)
        try:
            interaction = self.operator.process_interaction()
        finally:
            self.operator.delete_last_interaction()
        perception = self.operator.process_perception(images=images)
        return {
            "prompt": interaction["processed_prompt"],
            "images": perception["images"],
        }

    def __call__(
        self,
        prompt: str = "",
        images=None,
        output_path: Optional[str] = None,
        fps: Optional[int] = None,
        return_dict: bool = False,
        **kwargs,
    ):
        """Execute the complete pipeline generation flow."""
        if self.synthesis_model is None:
            raise RuntimeError("Synthesis model is not loaded. Use from_pretrained() first.")

        runtime_overrides: dict[str, Any] = {}
        for key in tuple(kwargs):
            target_key = _CALL_RUNTIME_ALIASES.get(key, key)
            if target_key in _RUNTIME_KWARGS:
                runtime_overrides[target_key] = kwargs.pop(key)

        # Apply per-call overrides to the long-lived synthesis state, then restore
        # them after predict so one call's num_frames/ddim_steps cannot leak into
        # the next call on the same pipeline instance.
        _missing = object()
        saved_runtime_kwargs: dict[str, Any] | None = None
        saved_generator_attrs: dict[str, Any] | None = None
        generator = None
        if runtime_overrides:
            runtime_kwargs = getattr(self.synthesis_model, "runtime_kwargs", None)
            if isinstance(runtime_kwargs, dict):
                saved_runtime_kwargs = {key: runtime_kwargs.get(key, _missing) for key in runtime_overrides}
                runtime_kwargs.update(runtime_overrides)
            generator = getattr(self.synthesis_model, "generator", None)
            if generator is not None:
                saved_generator_attrs = {key: getattr(generator, key, _missing) for key in runtime_overrides}
                for key, value in runtime_overrides.items():
                    setattr(generator, key, value)

        try:
            processed = self.process(prompt=prompt, images=images)
            result = self.synthesis_model.predict(
                prompt=processed["prompt"],
                images=processed["images"],
                output_path=output_path,
                fps=fps,
                return_dict=True,
                **kwargs,
            )
        finally:
            if saved_generator_attrs and getattr(self.synthesis_model, "generator", None) is generator:
                for key, value in saved_generator_attrs.items():
                    if value is _missing:
                        try:
                            delattr(generator, key)
                        except AttributeError:
                            pass
                    else:
                        setattr(generator, key, value)
            if saved_runtime_kwargs is not None:
                runtime_kwargs = getattr(self.synthesis_model, "runtime_kwargs", None)
                if isinstance(runtime_kwargs, dict):
                    for key, value in saved_runtime_kwargs.items():
                        if value is _missing:
                            runtime_kwargs.pop(key, None)
                        else:
                            runtime_kwargs[key] = value

        if return_dict:
            return result
        return result["video"]

    def stream(
        self,
        prompt: str = "",
        images=None,
        output_path: Optional[str] = None,
        fps: Optional[int] = None,
        return_dict: bool = False,
        **kwargs,
    ):
        """Stream visual generation outputs chunk by chunk."""
        if self.memory_module is None:
            raise ValueError("memory_module is not initialized")

        current_images = images
        if current_images is None and self.generation_type == "i2v":
            current_images = self.memory_module.select(prefer_type="image")
            if current_images is None:
                raise ValueError("stream() for i2v models requires an initial image on the first call.")
        elif current_images is not None:
            self.memory_module.record(current_images, metadata={"kind": "input_image"})

        result = self(
            prompt=prompt,
            images=current_images,
            output_path=output_path,
            fps=fps,
            return_dict=True,
            **kwargs,
        )
        self.memory_module.record(
            result["video"],
            metadata={
                "prompt": prompt,
                "model_name": self.model_name,
                "generation_type": self.generation_type,
            },
        )
        if return_dict:
            return result
        return result["video"]

    def get_operator(self) -> RuntimeVideoOperator:
        """Get operator for VideoCrafterPipelineBase."""
        return self.operator

    def get_synthesis_model(self):
        """Get synthesis model for VideoCrafterPipelineBase."""
        return self.synthesis_model


__all__ = ["VideoCrafterPipelineBase"]
