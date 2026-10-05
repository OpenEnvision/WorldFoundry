"""HunyuanWorld-Voyager pipeline backed by a local official checkout."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from worldfoundry.core.io import artifact_root_path
from worldfoundry.pipelines.pipeline_utils import PipelineABC
from worldfoundry.synthesis.visual_generation.hunyuan_world.voyager_official import VoyagerOfficialRuntime

if TYPE_CHECKING:
    from worldfoundry.core.contracts import PipelineInvocation


class HunyuanWorldVoyagerOfficialPipeline(PipelineABC):
    MODEL_ID = "hunyuanworld-voyager"

    def __init__(self, *, runtime: VoyagerOfficialRuntime | None = None, device: str = "cuda") -> None:
        self.model_id = self.MODEL_ID
        self.runtime = runtime or VoyagerOfficialRuntime(device=device)
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        **kwargs: Any,
    ) -> "HunyuanWorldVoyagerOfficialPipeline":
        options = dict(model_path) if isinstance(model_path, Mapping) else {}
        if model_path is not None and not isinstance(model_path, Mapping):
            options["model_path"] = model_path
        options.update(dict(required_components or {}))
        options.update(kwargs)
        runtime_options = {
            key: options[key]
            for key in ("source_root", "checkpoint_root", "python_executable")
            if options.get(key) is not None
        }
        if "checkpoint_root" not in runtime_options:
            for key in ("model_path", "pretrained_model_path", "repo_root"):
                path = options.get(key)
                if isinstance(path, (str, Path)) and Path(path).expanduser().is_dir():
                    runtime_options["checkpoint_root"] = path
                    break
        return cls(runtime=VoyagerOfficialRuntime(device=device, **runtime_options), device=device)

    def __call__(
        self,
        prompt: str,
        images: Any = None,
        video: Any = None,
        actions: Any = None,
        condition_dir: str | Path | None = None,
        output_path: str | Path | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        if video is not None:
            raise ValueError("Voyager official inference does not accept a video input")
        if actions is not None and not (isinstance(actions, (str, bytes, tuple, list, dict, set)) and len(actions) == 0):
            raise ValueError("Camera actions must be baked into the Voyager condition directory")
        if condition_dir is None and isinstance(images, (str, Path)) and Path(images).is_dir():
            condition_dir = images
        if condition_dir is None:
            raise ValueError("Voyager requires a 49-frame official condition_dir; raw-image preprocessing is separate")
        target = (
            Path(output_path) if output_path else artifact_root_path() / "pipeline_eval" / "hunyuanworld-voyager.mp4"
        )
        result = self.runtime.predict(prompt=prompt, condition_dir=condition_dir, output_path=target, **kwargs)
        if return_dict:
            return result
        if result["status"] != "succeeded":
            reason = result.get("error") or "; ".join(result.get("blocked_reasons", []))
            raise RuntimeError(reason or "Voyager official inference failed")
        return result["artifact_path"]

    def run_pipeline_invocation(self, invocation: PipelineInvocation) -> Mapping[str, Any]:
        return self(
            prompt=invocation.prompt,
            images=invocation.image,
            video=invocation.video,
            actions=invocation.interactions,
            condition_dir=invocation.operator_kwargs.get("condition_dir"),
            output_path=invocation.output_path,
            return_dict=True,
            **dict(invocation.pipeline_kwargs),
        )

    def get_synthesis_model(self) -> VoyagerOfficialRuntime:
        return self.runtime


__all__ = ["HunyuanWorldVoyagerOfficialPipeline"]
