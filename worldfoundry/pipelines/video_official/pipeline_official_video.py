"""Official Video visual generation pipeline module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from worldfoundry.core.io import artifact_root_path
from worldfoundry.core.media.codecs.image import load_pil_image, materialize_image_input
from worldfoundry.operators.runtime_video_operator import RuntimeVideoOperator
from worldfoundry.pipelines.pipeline_utils import PipelineABC
from worldfoundry.synthesis.visual_generation.memory.video import VideoArtifactMemory
from worldfoundry.synthesis.visual_generation.official_video_runtime import (
    OfficialVideoRuntime,
)

if TYPE_CHECKING:
    from worldfoundry.core.contracts import PipelineInvocation


class OfficialVideoPipeline(PipelineABC):
    """WorldFoundry pipeline for data-backed official video model runtimes."""

    MODEL_ID = ""
    GENERATION_TYPE = "t2v"

    def __init__(
        self,
        *,
        model_id: str | None = None,
        runtime: OfficialVideoRuntime | None = None,
        operator: RuntimeVideoOperator | None = None,
        memory_module: VideoArtifactMemory | None = None,
        device: str = "cuda",
    ) -> None:
        """Initialize the pipeline and configure runtime components."""
        self.model_id = model_id or self.MODEL_ID
        self.runtime = runtime or OfficialVideoRuntime.from_model_id(self.model_id, device=device)
        self.operator = operator or RuntimeVideoOperator(generation_type=self.GENERATION_TYPE)
        self.memory_module = memory_module or VideoArtifactMemory(model_id=self.model_id)
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "OfficialVideoPipeline":
        """Load the pipeline from pretrained checkpoints and configurations."""
        resolved_model_id = cls.MODEL_ID or model_id
        runtime_kwargs: dict[str, Any] = {}
        if isinstance(model_path, Mapping):
            runtime_kwargs.update(model_path)
        elif model_path:
            runtime_kwargs["checkpoint_path"] = str(model_path)
        runtime_kwargs.update(dict(required_components or {}))
        runtime_kwargs.update(kwargs)
        for key in (*PipelineABC.FRAMEWORK_LOADING_OPTION_KEYS, "model_id", "pipeline_target", "runtime_profile"):
            runtime_kwargs.pop(key, None)
        runtime = OfficialVideoRuntime.from_model_id(resolved_model_id, device=device, **runtime_kwargs)
        runtime.prepare()
        return cls(model_id=resolved_model_id, runtime=runtime, device=device)

    def process(self, prompt: str, images: Any = None, video: Any = None, **kwargs: Any) -> dict[str, Any]:
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
            "video": video,
        }

    def _materialize_image(self, images: Any, output_path: Path) -> str | None:
        """Materialize image for OfficialVideoPipeline."""
        if images is None:
            return None
        temp_dir = output_path.parent / f".{output_path.stem}_inputs"
        # Ensure target directory hierarchy exists before writing outputs
        temp_dir.mkdir(parents=True, exist_ok=True)
        return materialize_image_input(load_pil_image(images), temp_dir, filename="input.png")

    def __call__(
        self,
        prompt: str,
        images: Any = None,
        video: Any = None,
        num_frames: int | None = None,
        fps: int | None = None,
        output_path: str | Path | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Execute the complete pipeline generation flow."""
        target = Path(output_path) if output_path else artifact_root_path() / "pipeline_eval" / f"{self.model_id}.mp4"
        processed = self.process(prompt=prompt, images=images, video=video)
        runtime_kwargs = dict(kwargs)
        if num_frames is not None:
            runtime_kwargs.setdefault("num_frames", int(num_frames))
        if fps is not None:
            runtime_kwargs.setdefault("fps", int(fps))
        explicit_image_path = runtime_kwargs.pop("image_path", None)
        explicit_video_path = runtime_kwargs.pop("video_path", None)
        image_path = self._materialize_image(processed["images"], target) or explicit_image_path
        video_path = processed["video"] or explicit_video_path
        result = self.runtime.generate(
            prompt=processed["prompt"],
            image_path=image_path,
            video_path=video_path,
            output_path=target,
            **runtime_kwargs,
        )
        if return_dict:
            return result
        if isinstance(result, Mapping) and result.get("status") == "failed":
            message = result.get("error") or result.get("blocked_reason") or f"{self.model_id} official runtime failed"
            raise RuntimeError(str(message))
        return result.get("artifact_path") or result

    def run_pipeline_invocation(self, invocation: PipelineInvocation) -> Mapping[str, Any]:
        """Run pipeline invocation for OfficialVideoPipeline."""
        kwargs = dict(invocation.pipeline_kwargs)
        kwargs.update({"operator_kwargs": dict(invocation.operator_kwargs)})
        return self(
            prompt=invocation.prompt,
            images=invocation.image,
            video=invocation.video,
            output_path=invocation.output_path,
            return_dict=True,
            **kwargs,
        )

    def stream(self, prompt: str, images: Any = None, **kwargs: Any) -> Any:
        """Stream visual generation outputs chunk by chunk."""
        result = self(prompt=prompt, images=images, return_dict=True, **kwargs)
        self.memory_module.record(
            result.get("artifact_path") or result,
            metadata={"prompt": prompt, "model_name": self.model_id, "generation_type": self.GENERATION_TYPE},
        )
        return result.get("artifact_path") or result

    def get_operator(self) -> RuntimeVideoOperator:
        """Get operator for OfficialVideoPipeline."""
        return self.operator

    def get_synthesis_model(self) -> OfficialVideoRuntime:
        """Get synthesis model for OfficialVideoPipeline."""
        return self.runtime


class FramePackPipeline(OfficialVideoPipeline):
    """Pipeline implementation for FramePack visual generation."""
    MODEL_ID = "framepack"
    GENERATION_TYPE = "i2v"


class I2VGenXLPipeline(OfficialVideoPipeline):
    """Pipeline implementation for I2VGenXL visual generation."""
    MODEL_ID = "i2vgen-xl"
    GENERATION_TYPE = "i2v"


class MAGI1Pipeline(OfficialVideoPipeline):
    """Pipeline implementation for MAGI1 visual generation."""
    MODEL_ID = "magi-1"
    GENERATION_TYPE = "i2v"


class Mochi1PreviewT2VPipeline(OfficialVideoPipeline):
    """Pipeline implementation for Mochi1PreviewT2V visual generation."""
    MODEL_ID = "mochi-1-preview-t2v"
    GENERATION_TYPE = "t2v"


class ModelScopeT2VPipeline(OfficialVideoPipeline):
    """Pipeline implementation for ModelScopeT2V visual generation."""
    MODEL_ID = "modelscope-t2v"
    GENERATION_TYPE = "t2v"


class OpenSoraPlanPipeline(OfficialVideoPipeline):
    """Pipeline implementation for OpenSoraPlan visual generation."""
    MODEL_ID = "open-sora-plan"
    GENERATION_TYPE = "t2v"


class OpenSoraPipeline(OfficialVideoPipeline):
    """Pipeline implementation for OpenSora visual generation."""
    MODEL_ID = "open-sora"
    GENERATION_TYPE = "t2v"


class KreaRealtimeVideoPipeline(OfficialVideoPipeline):
    """Pipeline implementation for KreaRealtimeVideo visual generation."""
    MODEL_ID = "krea-realtime-video"
    GENERATION_TYPE = "t2v"


class MMAudioPipeline(OfficialVideoPipeline):
    """Pipeline implementation for MMAudio visual generation."""
    MODEL_ID = "mmaudio"
    GENERATION_TYPE = "v2a"


class OmniVinciPipeline(OfficialVideoPipeline):
    """Pipeline implementation for OmniVinci visual generation."""
    MODEL_ID = "omnivinci"
    GENERATION_TYPE = "multimodal"


class Qwen25OmniPipeline(OfficialVideoPipeline):
    """Pipeline implementation for Qwen25Omni visual generation."""
    MODEL_ID = "qwen2.5-omni"
    GENERATION_TYPE = "multimodal"


class SAMA14BPipeline(OfficialVideoPipeline):
    """Pipeline implementation for SAMA14B visual generation."""
    MODEL_ID = "sama-14b"
    GENERATION_TYPE = "v2v"

    def __call__(
        self,
        prompt: str,
        images: Any = None,
        video: Any = None,
        num_frames: int | None = None,
        fps: int | None = None,
        output_path: str | Path | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        unsupported_conditions = (
            "image",
            "image_path",
            "mask",
            "mask_path",
            "mask_image",
            "mask_video",
            "reference",
            "reference_image",
            "reference_image_path",
            "ref_image_path",
        )
        unsupported = [name for name in unsupported_conditions if kwargs.get(name) is not None]
        operator_kwargs = kwargs.get("operator_kwargs")
        if isinstance(operator_kwargs, Mapping):
            unsupported.extend(
                f"operator_kwargs.{name}"
                for name in unsupported_conditions
                if operator_kwargs.get(name) is not None
            )
        if images is not None:
            unsupported.append("images")
        if unsupported:
            raise ValueError(
                "SAMA-14B's released inference route accepts a source video and text prompt, "
                f"not {', '.join(unsupported)}."
            )
        return super().__call__(
            prompt=prompt,
            video=video,
            num_frames=num_frames,
            fps=fps,
            output_path=output_path,
            return_dict=return_dict,
            **kwargs,
        )

    def run_pipeline_invocation(self, invocation: PipelineInvocation) -> Mapping[str, Any]:
        if invocation.ref_image_path not in (None, ""):
            raise ValueError(
                "SAMA-14B's released inference route does not accept a separate reference image."
            )
        return super().run_pipeline_invocation(invocation)


class SpatialLadderPipeline(OfficialVideoPipeline):
    """Pipeline implementation for SpatialLadder visual generation."""
    MODEL_ID = "spatial-ladder"
    GENERATION_TYPE = "multimodal"


class SpatialReasonerPipeline(OfficialVideoPipeline):
    """Pipeline implementation for SpatialReasoner visual generation."""
    MODEL_ID = "spatial-reasoner"
    GENERATION_TYPE = "multimodal"


class UniAnimateDiTPipeline(OfficialVideoPipeline):
    """Pipeline implementation for UniAnimateDiT visual generation."""
    MODEL_ID = "unianimate-dit"
    GENERATION_TYPE = "i2v"


__all__ = [
    "FramePackPipeline",
    "I2VGenXLPipeline",
    "KreaRealtimeVideoPipeline",
    "MAGI1Pipeline",
    "MMAudioPipeline",
    "Mochi1PreviewT2VPipeline",
    "ModelScopeT2VPipeline",
    "OmniVinciPipeline",
    "OpenSoraPipeline",
    "OpenSoraPlanPipeline",
    "OfficialVideoPipeline",
    "Qwen25OmniPipeline",
    "SAMA14BPipeline",
    "SpatialLadderPipeline",
    "SpatialReasonerPipeline",
    "UniAnimateDiTPipeline",
]
