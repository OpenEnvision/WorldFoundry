"""Pusa Vidgen visual generation pipeline module."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from worldfoundry.runtime.env import resolve_hfd_root
from worldfoundry.synthesis.visual_generation.pusa_vidgen.adapter import PusaVidGenRuntime

from ..pipeline_utils import PipelineABC

_HF_ROOT = resolve_hfd_root()
_DEFAULT_CKPT_ROOT = _HF_ROOT / "RaphaelLiu--Pusa-Wan2.2-V1"
_DEFAULT_BASE_MODEL_ROOT = _HF_ROOT / "Wan-AI--Wan2.2-T2V-A14B"
_DEFAULT_LIGHTX2V_ROOT = _HF_ROOT / "lightx2v--Wan2.2-Lightning"


def _options_from(
    model_path: Any, required_components: dict[str, Any] | None, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Merge shared loader inputs into model-specific adapter options."""
    options: dict[str, Any] = {}
    if isinstance(model_path, dict):
        options.update(model_path)
    elif model_path is not None:
        options["checkpoint_root"] = model_path
    options.update(required_components or {})
    options.update(kwargs)
    return options


def _json_safe(value: Any) -> Any:
    """Convert nested path-like values before writing JSON reports."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return value


class PusaVidGenPipeline(PipelineABC):
    """In-tree Pusa VidGen pipeline with no runtime dependency on external source checkouts."""

    MODEL_ID = "pusa-vidgen"

    def __init__(
        self,
        runtime: PusaVidGenRuntime,
        model_id: str | None = None,
    ) -> None:
        """
        Create a Pusa VidGen pipeline around the internal runtime shim.

        Args:
            runtime: In-tree lightweight Pusa runtime adapter.
            model_id: Optional WorldFoundry model identifier override.
        """
        self.model_id = model_id or self.MODEL_ID
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
    ) -> PusaVidGenPipeline:
        """
        Build Pusa VidGen from internal code and local checkpoint metadata only.

        Args:
            model_path: Optional checkpoint root or component dictionary.
            required_components: Optional model-specific paths.
            device: Runtime device label.
            model_id: Accepted for the shared loader signature.
            **kwargs: Additional adapter options.
        """
        options = _options_from(model_path, required_components, kwargs)
        resolved_model_id = str(options.pop("model_id", model_id or cls.MODEL_ID))
        python_executable = options.pop("python_executable", None)
        python_env_dir = options.pop("python_env_dir", options.pop("conda_dir", None))
        if python_executable is None and python_env_dir is not None:
            python_executable = Path(python_env_dir).expanduser() / "bin" / "python"
        runtime = PusaVidGenRuntime(
            model_id=resolved_model_id,
            checkpoint_root=options.pop("checkpoint_root", options.pop("lora_root", _DEFAULT_CKPT_ROOT)),
            base_model_root=options.pop("base_model_root", options.pop("wan22_root", _DEFAULT_BASE_MODEL_ROOT)),
            high_lora_path=options.pop("high_lora_path", None),
            low_lora_path=options.pop("low_lora_path", None),
            lightx2v_root=options.pop("lightx2v_root", _DEFAULT_LIGHTX2V_ROOT),
            device=device,
            python_executable=python_executable,
            runner_version=options.pop("runner_version", None),
        )
        return cls(runtime=runtime, model_id=resolved_model_id)

    def process(
        self,
        prompt: str | None = None,
        images: Any = None,
        video: Any = None,
        interactions: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Prepare an internal Pusa VidGen request for later execution.

        Args:
            prompt: Text prompt for generation.
            images: Optional conditioning image path or paths.
            video: Optional conditioning video path.
            interactions: Optional conditioning frame positions.
            **kwargs: Model-specific generation settings.
        """
        return self.runtime.prepare(prompt=prompt, images=images, video=video, interactions=interactions, **kwargs)

    def __call__(
        self,
        prompt: str | None = None,
        images: Any = None,
        video: Any = None,
        interactions: Any = None,
        output_path: str | Path | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | str:
        """
        Run Pusa VidGen through the real in-tree public runner.

        Args:
            prompt: Text prompt for generation.
            images: Optional conditioning image path or paths.
            video: Optional conditioning video path.
            interactions: Optional conditioning frame positions.
            output_path: Optional JSON artifact path.
            return_dict: Return structured metadata instead of artifact path.
            **kwargs: Model-specific generation settings.
        """
        execute = bool(kwargs.pop("execute", False))
        timeout_seconds = int(kwargs.pop("timeout_seconds", 1800))
        requested_output_dir = kwargs.pop("output_dir", None)
        request = self.process(prompt=prompt, images=images, video=video, interactions=interactions, **kwargs)
        if not execute:
            raise RuntimeError("Pusa VidGen requires execute=True; request-plan artifacts are no longer emitted.")

        target = (
            Path(output_path).expanduser().resolve()
            if output_path is not None
            else Path.cwd() / "pusa_vidgen_public_runner_result.json"
        )
        if target.suffix.lower() == ".mp4":
            artifact_path = target
            report_path = target.with_suffix(".json")
            run_output_dir = (
                Path(requested_output_dir).expanduser().resolve()
                if requested_output_dir is not None
                else artifact_path.parent
            )
        else:
            report_path = target
            run_output_dir = (
                Path(requested_output_dir).expanduser().resolve()
                if requested_output_dir is not None
                else report_path.parent / "outputs"
            )
            artifact_path = run_output_dir / "pusa_vidgen_public_runner.mp4"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        run_output_dir.mkdir(parents=True, exist_ok=True)
        runtime_plan = self.runtime.build_plan(
            request=request,
            output_dir=run_output_dir,
            output_path=artifact_path,
        )
        run_result = self.runtime.run_plan(
            runtime_plan,
            timeout_seconds=timeout_seconds,
            log_dir=report_path.parent,
        )
        generated_files = list(run_result.get("generated_files") or [])
        resolved_artifact_path = generated_files[0] if generated_files else str(artifact_path)
        result = {
            "status": run_result.get("status", "failed"),
            "ok": bool(run_result.get("ok")),
            "model_id": self.model_id,
            "artifact_kind": "generated_video",
            "artifact_path": resolved_artifact_path,
            "artifact_files": generated_files,
            "runtime": "worldfoundry.synthesis.visual_generation.pusa_vidgen.pusa_v1_runner",
            "backend_quality": "public_runner_official_runner",
            "metadata_path": run_result.get("metadata_path") or str(report_path),
            "request": request,
            "runtime_result": run_result,
        }
        result = _json_safe(result)
        report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        result["report_path"] = str(report_path)
        self.history.append(result)
        if return_dict:
            return result
        return str(resolved_artifact_path)
