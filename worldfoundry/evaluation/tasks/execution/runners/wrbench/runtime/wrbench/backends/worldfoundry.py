"""Real generation through WorldFoundry's canonical in-tree model runner."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from wrbench.backends.base import GenerationRequest, GenerationResult
from wrbench.registry import model_record
from wrbench.runtime import RuntimeConfig


class WorldFoundryPipelineBackend:
    """Delegate WRBench camera payloads to the shared WorldFoundry pipeline catalog."""

    name = "worldfoundry_pipeline"

    def __init__(self, runtime: RuntimeConfig | None = None) -> None:
        self.runtime = runtime

    @staticmethod
    def _worldfoundry_model_id(model: str) -> str:
        return model_record(model).worldfoundry_model_id

    def available(self) -> tuple[bool, str]:
        try:
            from worldfoundry.evaluation.models.catalog.zoo_registry import load_model_zoo_registry

            load_model_zoo_registry()
        except Exception as exc:  # noqa: BLE001 - availability must return diagnostics.
            return False, f"WorldFoundry model catalog is unavailable: {type(exc).__name__}: {exc}"
        return True, "WorldFoundry in-tree model catalog is available"

    def available_for(self, model: str) -> tuple[bool, str]:
        model_id = self._worldfoundry_model_id(model)
        try:
            from worldfoundry.evaluation.models.catalog.zoo_registry import load_model_zoo_registry

            entry = load_model_zoo_registry().get(model_id)
        except KeyError:
            return False, (
                f"WRBench model {model!r} does not resolve to an in-tree WorldFoundry "
                f"model catalog entry ({model_id!r})"
            )
        except Exception as exc:  # noqa: BLE001 - availability must return diagnostics.
            return False, f"WorldFoundry model catalog lookup failed: {type(exc).__name__}: {exc}"
        if not entry.runner_target:
            return False, f"WorldFoundry model {entry.model_id!r} has no runnable in-tree runner"
        return True, f"WorldFoundry model {entry.model_id!r} uses {entry.runner_target}"

    def _runner_overrides(self, model: str, output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        parameters: dict[str, Any] = {}
        runtime: dict[str, Any] = {"output_dir": str(output_dir)}
        configured = self.runtime.model(model) if self.runtime is not None else None
        if configured is None:
            return parameters, runtime
        if configured.model_path:
            parameters["model_path"] = configured.model_path
        required_components = dict(configured.extra_paths)
        required_components.setdefault("python_executable", configured.python_bin)
        if configured.env:
            required_components["env"] = dict(configured.env)
        if required_components:
            parameters["required_components"] = required_components
        runtime["device"] = f"cuda:{configured.gpu_id}"
        return parameters, runtime

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {str(key): WorldFoundryPipelineBackend._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [WorldFoundryPipelineBackend._json_safe(item) for item in value]
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            return WorldFoundryPipelineBackend._json_safe(tolist())
        return value

    @staticmethod
    def _worldfoundry_request(request: GenerationRequest):
        from worldfoundry.evaluation.api import GenerationRequest as WorldFoundryGenerationRequest

        record = model_record(request.model)
        inputs: dict[str, Any] = {"prompt": request.prompt}
        if request.image_path is not None:
            inputs["image"] = str(request.image_path)
        if request.source_video_path is not None:
            inputs["video"] = str(request.source_video_path)

        camera_script = str(request.extra.get("camera_script") or "")
        controls = {
            "camera_script": camera_script,
            "camera_payload_type": request.payload.payload_type,
            "camera_payload": WorldFoundryPipelineBackend._json_safe(request.payload.payload),
            "target_camera_poses": WorldFoundryPipelineBackend._json_safe(
                request.payload.target_trajectory.to_c2w()
            ),
        }
        generation_kwargs = WorldFoundryPipelineBackend._json_safe(dict(request.payload.payload))
        generation_kwargs.setdefault("num_frames", request.payload.target_trajectory.frame_count)
        generation_kwargs.setdefault("fps", request.payload.target_trajectory.fps)
        generation_kwargs.setdefault("width", record.default_width)
        generation_kwargs.setdefault("height", record.default_height)

        # Some in-tree interactive pipelines consume a normalized interaction
        # sequence directly. Preserve model-native action payloads when the
        # WRBench adapter exposes one, without inventing a second model adapter.
        for key in ("interactions", "action_list", "actions", "wasd_action", "pose"):
            value = request.payload.payload.get(key)
            if value is not None:
                controls["interactions"] = value
                break

        return WorldFoundryGenerationRequest(
            sample_id=request.output_path.stem,
            task_name="wrbench",
            inputs=inputs,
            controls=controls,
            generation_kwargs=generation_kwargs,
            output_schema={"generated_video": {"kind": "video"}},
        )

    @staticmethod
    def _materialize_output(result: Any, requested_output: Path) -> Path:
        from worldfoundry.evaluation.api import local_path_for_uri

        if requested_output.is_file():
            return requested_output
        preferred_names = ("generated_video", "video", "generated_world")
        artifacts = dict(result.artifacts or {})
        artifact = next((artifacts[name] for name in preferred_names if name in artifacts), None)
        if artifact is None and artifacts:
            artifact = next(iter(artifacts.values()))
        if artifact is None:
            raise FileNotFoundError("WorldFoundry runner reported success without an output artifact")
        source = local_path_for_uri(artifact.uri)
        if source is None or not source.is_file():
            raise FileNotFoundError(f"WorldFoundry output artifact is not a local file: {artifact.uri}")
        requested_output.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != requested_output.resolve():
            shutil.copy2(source, requested_output)
        return requested_output

    def generate(self, request: GenerationRequest) -> GenerationResult:
        ok, reason = self.available_for(request.model)
        if not ok:
            return GenerationResult(success=False, message=reason)

        from worldfoundry.evaluation.api import is_generation_result_successful
        from worldfoundry.evaluation.models import resolve_model_zoo_runner
        from worldfoundry.evaluation.utils import MODEL_ZOO_DIR

        model_id = self._worldfoundry_model_id(request.model)
        parameters, runtime = self._runner_overrides(request.model, request.output_path.parent)
        resolved = None
        try:
            resolved = resolve_model_zoo_runner(
                model_id,
                manifest_dir=MODEL_ZOO_DIR,
                parameters=parameters,
                runtime=runtime,
            )
            results = list(resolved.runner.generate([self._worldfoundry_request(request)]))
            if len(results) != 1:
                return GenerationResult(
                    success=False,
                    message=f"WorldFoundry runner returned {len(results)} results for one WRBench request",
                )
            result = results[0]
            if not is_generation_result_successful(result):
                return GenerationResult(
                    success=False,
                    message=result.error or f"WorldFoundry generation failed with status {result.status!r}",
                )
            output_path = self._materialize_output(result, request.output_path)
            return GenerationResult(
                success=True,
                output_path=output_path,
                message=f"generated through WorldFoundry model {resolved.model_id!r}",
                artifacts={name: artifact.uri for name, artifact in result.artifacts.items()},
            )
        except Exception as exc:  # noqa: BLE001 - backend returns structured failure.
            return GenerationResult(
                success=False,
                message=f"WorldFoundry generation failed: {type(exc).__name__}: {exc}",
            )
        finally:
            if resolved is not None:
                cleanup = getattr(resolved.runner, "cleanup", None)
                if callable(cleanup):
                    cleanup()


__all__ = ["WorldFoundryPipelineBackend"]
