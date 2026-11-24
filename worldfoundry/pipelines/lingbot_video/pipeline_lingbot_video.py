"""WorldFoundry pipeline for the in-tree LingBot-Video inference runtime."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.core.io.paths import resolve_local_hf_model_path
from worldfoundry.pipelines.pipeline_utils import PipelineABC
from worldfoundry.runtime.env import resolve_hfd_root
from worldfoundry.synthesis.visual_generation.lingbot_video import LingBotVideoRuntime

_HFD_ROOT = resolve_hfd_root()


def _local_checkpoint(repo_id: str) -> Path:
    try:
        return resolve_local_hf_model_path(repo_id, required_files=("model_index.json",))
    except FileNotFoundError:
        return _HFD_ROOT / repo_id.replace("/", "--")


_CHECKPOINTS = {
    "dense": _local_checkpoint("robbyant/lingbot-video-dense-1.3b"),
    "moe": _local_checkpoint("robbyant/lingbot-video-moe-30b-a3b"),
}


def _options_from(
    model_path: Any, required_components: dict[str, Any] | None, kwargs: dict[str, Any]
) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if isinstance(model_path, dict):
        options.update(model_path)
    elif model_path is not None:
        options["checkpoint_dir"] = model_path
    options.update(required_components or {})
    options.update(kwargs)
    return options


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return value


class LingBotVideoGenerationPipeline(PipelineABC):
    """Execute Dense or MoE LingBot-Video T2I, T2V, and TI2V inference."""

    MODEL_ID = "lingbot-video"

    def __init__(
        self,
        runtime: LingBotVideoRuntime,
        *,
        model_id: str | None = None,
        variant: str = "dense",
    ) -> None:
        self.model_id = model_id or self.MODEL_ID
        self.variant = variant
        self.runtime = runtime
        self.history: list[dict[str, Any]] = []

    @classmethod
    def from_pretrained(
        cls,
        model_path: Any = None,
        required_components: dict[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> LingBotVideoGenerationPipeline:
        options = _options_from(model_path, required_components, kwargs)
        resolved_model_id = str(options.pop("model_id", model_id or cls.MODEL_ID))
        variant = str(options.pop("variant", options.pop("variant_id", "dense"))).lower()
        if variant not in _CHECKPOINTS:
            raise ValueError(f"Unsupported LingBot-Video variant {variant!r}; expected dense or moe.")
        checkpoint_dir = options.pop("checkpoint_dir", options.pop("model_dir", _CHECKPOINTS[variant]))
        python_executable = options.pop("python_executable", None)
        python_env_dir = options.pop("python_env_dir", options.pop("conda_dir", None))
        if python_executable is None and python_env_dir is not None:
            python_executable = Path(python_env_dir).expanduser() / "bin" / "python"
        runtime = LingBotVideoRuntime(
            checkpoint_dir=checkpoint_dir,
            device=device,
            python_executable=python_executable,
        )
        return cls(runtime, model_id=resolved_model_id, variant=variant)

    def preflight(
        self,
        *,
        run_refiner: bool = False,
        refiner_model_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        return self.runtime.preflight(
            run_refiner=run_refiner,
            refiner_model_dir=refiner_model_dir,
        )

    def process(
        self,
        prompt: str | None = None,
        images: Any = None,
        video: Any = None,
        interactions: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del video, interactions
        request = dict(kwargs)
        request["prompt"] = prompt
        request["images"] = images
        request.setdefault("mode", "ti2v" if images is not None else "t2v")
        request.setdefault("variant", self.variant)
        return request

    def __call__(
        self,
        prompt: str | None = None,
        images: Any = None,
        video: Any = None,
        interactions: Any = None,
        output_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        execute: bool = False,
        timeout_seconds: int = 7200,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | str:
        request = self.process(
            prompt=prompt,
            images=images,
            video=video,
            interactions=interactions,
            **kwargs,
        )
        if not execute:
            raise RuntimeError("LingBot-Video requires execute=True; request-plan artifacts are not emitted.")

        mode = str(request.get("mode") or "t2v").lower()
        media_suffix = ".png" if mode == "t2i" else ".mp4"
        requested = Path(output_path).expanduser().resolve() if output_path is not None else None
        if requested is not None and mode == "t2i" and requested.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            requested = requested.with_suffix(".png")
        elif requested is not None and mode != "t2i" and requested.suffix.lower() not in {".mp4", ".webm", ".mov"}:
            requested = requested.with_suffix(".mp4")
        if requested is not None and requested.suffix.lower() in {
            ".png",
            ".jpg",
            ".jpeg",
            ".mp4",
            ".webm",
            ".mov",
        }:
            artifact_path = requested
            report_path = requested.with_suffix(".json")
        else:
            report_path = requested or (Path.cwd() / "lingbot_video_result.json")
            artifact_root = Path(output_dir).expanduser().resolve() if output_dir is not None else report_path.parent
            artifact_path = artifact_root / f"lingbot_video{media_suffix}"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)

        run_refiner = bool(request.get("run_refiner") or request.get("refiner_model_dir"))
        preflight = self.preflight(
            run_refiner=run_refiner,
            refiner_model_dir=request.get("refiner_model_dir"),
        )
        if preflight["status"] != "ready":
            missing = [
                *preflight["missing_checkpoint_files"],
                *preflight["missing_runtime_files"],
            ]
            raise RuntimeError(f"LingBot-Video preflight failed; missing: {', '.join(missing)}")

        plan = self.runtime.build_plan(request=request, output_path=artifact_path)
        run_result = self.runtime.run_plan(plan, timeout_seconds=timeout_seconds, log_dir=report_path.parent)
        generated_files = list(run_result.get("generated_files") or [])
        resolved_artifact = (
            generated_files[-1]
            if run_refiner and generated_files
            else (generated_files[0] if generated_files else str(artifact_path))
        )
        result = _json_safe(
            {
                "status": run_result.get("status", "failed"),
                "ok": bool(run_result.get("ok")),
                "model_id": self.model_id,
                "variant": self.variant,
                "artifact_kind": ("generated_image" if mode == "t2i" else "generated_video"),
                "artifact_path": resolved_artifact,
                "artifact_files": generated_files,
                "runtime": "worldfoundry.synthesis.visual_generation.lingbot_video.runner",
                "backend_quality": "in_tree_official_inference",
                "metadata_path": str(report_path),
                "request": request,
                "preflight": preflight,
                "runtime_result": run_result,
            }
        )
        report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.history.append(result)
        if return_dict:
            return result
        return str(resolved_artifact)

    def stream(self, *args: Any, **kwargs: Any) -> dict[str, Any] | str:
        return self(*args, **kwargs)


__all__ = ["LingBotVideoGenerationPipeline"]
