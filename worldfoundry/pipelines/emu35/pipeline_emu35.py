"""WorldFoundry pipeline for BAAI Emu3.5's official text/image response."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from worldfoundry.core.io import artifact_root_path
from worldfoundry.pipelines.pipeline_utils import PipelineABC
from worldfoundry.synthesis.visual_generation.emu35 import Emu35Runtime

if TYPE_CHECKING:
    from worldfoundry.core.contracts import PipelineInvocation


class Emu35Pipeline(PipelineABC):
    MODEL_ID = "emu3.5"

    def __init__(self, *, runtime: Emu35Runtime | None = None, device: str = "cuda") -> None:
        self.model_id = self.MODEL_ID
        self.runtime = runtime or Emu35Runtime(device=device)
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        **kwargs: Any,
    ) -> "Emu35Pipeline":
        options = dict(model_path) if isinstance(model_path, Mapping) else {}
        if model_path is not None and not isinstance(model_path, Mapping):
            options["checkpoint_path"] = model_path
        options.update(dict(required_components or {}))
        options.update(kwargs)
        for key in (*PipelineABC.FRAMEWORK_LOADING_OPTION_KEYS, "pipeline_target", "runtime_profile"):
            options.pop(key, None)
        return cls(runtime=Emu35Runtime(device=device, **options), device=device)

    def __call__(
        self,
        prompt: str,
        images: Any = None,
        video: Any = None,
        output_path: str | Path | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        if video is not None:
            raise ValueError("Emu3.5 does not accept video input")
        target = Path(output_path) if output_path else artifact_root_path() / "pipeline_eval" / "emu3.5.pb"
        image_path = kwargs.pop("image_path", None)
        if images is not None:
            if image_path is not None:
                raise ValueError("Provide one Emu3.5 input image via images or image_path")
            if isinstance(images, (str, Path)) and Path(images).is_file():
                image_path = str(images)
            else:
                from worldfoundry.core.io.image import materialize_image_input

                image_path = materialize_image_input(images, target.parent / f".{target.stem}_inputs")
        result = self.runtime.predict(prompt=prompt, image_path=image_path, output_path=target, **kwargs)
        if return_dict:
            return result
        if result["status"] != "succeeded":
            reason = result.get("error") or "; ".join(result.get("blocked_reasons", []))
            raise RuntimeError(reason or "Emu3.5 inference failed")
        return result["artifact_path"]

    def run_pipeline_invocation(self, invocation: PipelineInvocation) -> Mapping[str, Any]:
        return self(
            prompt=invocation.prompt,
            images=invocation.image,
            video=invocation.video,
            output_path=invocation.output_path,
            return_dict=True,
            **dict(invocation.pipeline_kwargs),
        )

    def get_synthesis_model(self) -> Emu35Runtime:
        return self.runtime
