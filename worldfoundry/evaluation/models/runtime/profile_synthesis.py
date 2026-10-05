"""Synthesis bridge for runtime profiles (plan-only command-template synthesis).

Hosts :class:`RuntimeProfileSynthesis`, the bridge between runtime-profile
metadata and the :class:`~worldfoundry.synthesis.base_synthesis.BaseSynthesis`
interface. This module is imported lazily from
:mod:`worldfoundry.evaluation.models.runtime.profiles` so that plain profile
metadata loading never pays the ``worldfoundry.synthesis`` import cost
(which pulls torch when installed).

The bridge only *plans* runs: :meth:`RuntimeProfileSynthesis.predict` writes a
command plan JSON and returns ``"prepared"`` (``plan_only=True``) or
``"blocked"``; it never executes the planned command itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from worldfoundry.core.io.paths import scratch_directory
from worldfoundry.evaluation.models.runtime.profiles import (
    DEFAULT_COND_DIR,
    DEFAULT_SD15_ROOT,
    RuntimeProfile,
    _primary_source_context,
    _safe_name,
    load_runtime_profile,
    resolve_existing_tool,
)
from worldfoundry.evaluation.utils import REPO_ROOT
from worldfoundry.runtime.conda import resolve_conda_env_context, resolve_conda_executable
from worldfoundry.synthesis.base_synthesis import BaseSynthesis


def _coerce_path_input(value: Any, destination: Path, stem: str) -> str | None:
    """Coerce any PIL image, file path, or remote URL into a relative/absolute path string."""
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        text = str(value)
        if text.startswith(("http://", "https://", "s3://", "gs://", "hf://")):
            return text
        path = Path(text).expanduser()
        return str(path) if path.exists() else text
    save = getattr(value, "save", None)
    if callable(save):
        path = destination / f"{stem}.png"
        save(path)
        return str(path)
    return str(value)


def _json_safe(value: Any) -> Any:
    """Recursively convert custom/path objects so they are fully JSON-serializable."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())
    return str(value)


class RuntimeProfileSynthesis(BaseSynthesis):
    """Plan-only runtime-profile bridge for model-specific shims.

    A profile is runnable only when it has an in-tree backend implementation or
    an integrated command template. Metadata-only and profile-only records are
    provenance/planning surfaces and must not be treated as full integrations.

    ``predict`` never executes commands: it writes a command plan JSON and
    returns ``"prepared"`` (``plan_only=True``) or ``"blocked"`` (vendor
    runtime not yet ported in-tree).
    """

    MODEL_ID: str | None = None

    def __init__(
        self,
        profile: RuntimeProfile,
        *,
        device: str = "cuda",
        command_template: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize the synthesis shim with a runtime profile, device, command template, and environment."""
        self.profile = profile
        self.model_id = profile.model_id
        self.model_name = profile.model_id
        self.generation_type = "t2v" if profile.task_family == "video_generation" else "runtime_profile"
        self.device = device
        self.command_template = tuple(command_template or profile.command_template)
        self.env = dict(env or {})

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path=None,
        args=None,
        device=None,
        model_id: str | None = None,
        profile_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
        acquisition_root: str | Path | None = None,
        hf_models_root: str | Path | None = None,
        command_template: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> "RuntimeProfileSynthesis":
        """Load the runtime profile and instantiate a RuntimeProfileSynthesis instance."""
        del args
        options = dict(pretrained_model_path) if isinstance(pretrained_model_path, Mapping) else {}
        options.update(kwargs)
        resolved_model_id = str(options.get("model_id") or options.get("profile_id") or model_id or cls.MODEL_ID or "")
        if not resolved_model_id:
            raise ValueError("RuntimeProfileSynthesis requires model_id/profile_id.")
        profile = load_runtime_profile(
            resolved_model_id,
            manifest_path=manifest_path or options.get("manifest_path"),
            profile_path=profile_path or options.get("profile_path"),
            acquisition_root=acquisition_root or options.get("acquisition_root"),
            hf_models_root=hf_models_root or options.get("hf_models_root"),
        )
        resolved_template = command_template or options.get("command_template")
        return cls(
            profile=profile,
            device=str(device or options.get("device") or "cuda"),
            command_template=resolved_template,
            env=options.get("env"),
        )

    def _context(
        self,
        *,
        prompt: str,
        images: Any,
        video: Any,
        interactions: Sequence[str],
        output_path: str | Path | None,
        fps: int | None,
        run_dir: Path,
        extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Assemble formatting variables and inputs for the runtime command template.

        Writes prompt, actions, and extra_inputs to files under ``run_dir``
        and resolves conda/python paths for command substitution.

        Args:
            prompt: Text prompt for the generation request.
            images: Image input(s) — path, URL, or PIL image.
            video: Video input(s) — path, URL, or file-like object.
            interactions: Sequence of action/interaction strings.
            output_path: Desired output file path; defaults to the profile's
                ``artifact_filename`` inside ``run_dir``.
            fps: Frames-per-second override; included as ``""`` when unset.
            run_dir: Temporary working directory for the current run.
            extra: Arbitrary extra parameters forwarded into the template context.

        Returns:
            A ``dict[str, str]`` mapping template variable names to their
            resolved string values.
        """
        # ── Write input files ────────────────────────────
        prompt_path = run_dir / "prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        actions_path = run_dir / "actions.json"
        actions_path.write_text(
            json.dumps(_json_safe(list(interactions)), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        extra_inputs_path = run_dir / "extra_inputs.json"
        extra_inputs_path.write_text(
            json.dumps(_json_safe(dict(extra)), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # ── Resolve media inputs ──────────────────────────
        image_path = _coerce_path_input(images, run_dir, "input_image")
        video_path = _coerce_path_input(video, run_dir, "input_video")
        resolved_output = Path(output_path) if output_path is not None else run_dir / self.profile.artifact_filename
        if not resolved_output.is_absolute():
            resolved_output = (Path.cwd() / resolved_output).resolve()
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        # ── Resolve conda / python paths ──────────────────
        primary_ckpt = dict(self.profile.checkpoints[0]) if self.profile.checkpoints else {}
        source_context = _primary_source_context(self.profile.source_repos)
        conda_env = resolve_conda_env_context(self.model_id)
        conda_python = str(extra.get("python_executable") or "")
        if not conda_python:
            conda_python = str(conda_env.get("python_executable") or "")
            if not Path(conda_python).is_file():
                # NOTE: Fall back to the current interpreter if the env doesn't exist.
                conda_python = sys.executable
        conda_torchrun = str(extra.get("torchrun_executable") or "")
        if not conda_torchrun:
            conda_torchrun = resolve_conda_executable(self.model_id, "torchrun") or resolve_existing_tool("torchrun")
        conda_env_prefix = str(conda_env.get("env_prefix") or "")
        # ── Build the full template context ──────────────
        context = {
            "python": conda_python,
            "torchrun": conda_torchrun,
            "model_id": self.model_id,
            "display_name": self.profile.display_name,
            "worldfoundry_root": str(REPO_ROOT),
            "repo_root": str(REPO_ROOT),
            "checkpoint_dir": primary_ckpt.get("local_dir", ""),
            "prompt": prompt,
            "prompt_path": str(prompt_path),
            "actions_path": str(actions_path),
            "extra_inputs_path": str(extra_inputs_path),
            "image_path": image_path or "",
            "video_path": video_path or "",
            "output_path": str(resolved_output),
            "output_dir": str(resolved_output.parent),
            "run_dir": str(run_dir),
            "device": self.device,
            "fps": "" if fps is None else str(fps),
            "parallel": str(extra.get("parallel", 4)),
            "tensor_parallel_degree": str(extra.get("tensor_parallel_degree", 2)),
            "tp_degree": str(extra.get("tp_degree", extra.get("tensor_parallel_degree", 2))),
            "ulysses_degree": str(extra.get("ulysses_degree", 2)),
            "vae_url": str(extra.get("vae_url", "127.0.0.1")),
            "caption_url": str(extra.get("caption_url", "127.0.0.1")),
            "infer_steps": str(extra.get("infer_steps", 50)),
            "cfg_scale": str(extra.get("cfg_scale", 9.0)),
            "time_shift": str(extra.get("time_shift", 13.0)),
            "class_id": str(extra.get("class_id", 207)),
            "batch_size": str(extra.get("batch_size", 1)),
            "seed": str(extra.get("seed", 1234)),
            "height": str(extra.get("height", 256)),
            "width": str(extra.get("width", 256)),
            "condtype": str(extra.get("condtype", "both")),
            "cond_dir": str(extra.get("cond_dir", DEFAULT_COND_DIR)),
            "nproc_per_node": str(extra.get("nproc_per_node", 1)),
            "master_port": str(extra.get("master_port", 25000)),
            "config": str(extra.get("config", "")),
            "ckpt_path": str(extra.get("ckpt_path", primary_ckpt.get("local_dir", ""))),
            "unnorm_key": str(extra.get("unnorm_key", "bridge_orig")),
            "attn_implementation": str(extra.get("attn_implementation", "eager")),
            "torch_dtype": str(extra.get("torch_dtype", "auto")),
            "sd15_path": str(extra.get("sd15_path", DEFAULT_SD15_ROOT)),
            "motion_module_ckpt": str(extra.get("motion_module_ckpt", "")),
            "pose_adaptor_ckpt": str(extra.get("pose_adaptor_ckpt", primary_ckpt.get("local_dir", ""))),
            "trajectory_file": str(extra.get("trajectory_file", "")),
            "conda_env_name": str(conda_env.get("env_name") or ""),
            "conda_env_prefix": conda_env_prefix,
            "conda_env_exists": str(bool(conda_env.get("exists"))).lower(),
            "conda_env_driver_status": str(conda_env.get("driver_status") or ""),
            "conda_env_cuda_profile": str(conda_env.get("cuda_profile") or ""),
            "driver_cuda": str(conda_env.get("driver_cuda") or ""),
            "source_repos_json": json.dumps([dict(item) for item in self.profile.source_repos], ensure_ascii=False),
            "checkpoints_json": json.dumps([dict(item) for item in self.profile.checkpoints], ensure_ascii=False),
            "conda_env_json": json.dumps(conda_env, ensure_ascii=False),
            "backend_stage": self.profile.backend_stage,
            **source_context,
        }
        # Overlay any extra string/numeric parameters into the context.
        for key, value in extra.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                context[str(key)] = "" if value is None else str(value)
        return context

    def _format_command(self, context: Mapping[str, Any]) -> list[str]:
        """Format the profile command template with resolved execution context variables."""
        return [part.format(**context) for part in self.command_template]

    def predict(
        self,
        prompt: str = "",
        images: Any = None,
        video: Any = None,
        interactions: Sequence[str] = (),
        output_path: str | Path | None = None,
        fps: int | None = None,
        plan_only: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Plan the generation request using command-line templates.

        This method never executes the planned command. It writes a
        ``runtime_profile_plan.json`` and returns ``"prepared"`` when
        ``plan_only=True``, or ``"blocked"`` otherwise (the official vendor
        runtime is not yet ported in-tree).
        """
        # Accepted for backwards compatibility with callers that pass a
        # timeout; there is no execution path that could consume it.
        kwargs.pop("timeout_seconds", None)
        run_dir = Path(
            kwargs.pop("run_dir", "")
            or scratch_directory(f"{_safe_name(self.model_id)}_")
        )
        run_dir = run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        context = self._context(
            prompt=prompt,
            images=images,
            video=video,
            interactions=interactions,
            output_path=output_path,
            fps=fps,
            run_dir=run_dir,
            extra=kwargs,
        )
        command = self._format_command(context) if self.command_template else []
        plan_path = run_dir / "runtime_profile_plan.json"
        plan_payload = {
            "schema_version": "worldfoundry-runtime-profile-plan",
            "profile": self.profile.to_dict(),
            "context": context,
            "command": command,
        }
        plan_path.write_text(json.dumps(plan_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if plan_only:
            artifact_path = plan_path
            if output_path is not None:
                requested_output = Path(output_path).expanduser().resolve()
                if requested_output.suffix.lower() == ".json":
                    requested_output.parent.mkdir(parents=True, exist_ok=True)
                    requested_output.write_text(
                        json.dumps(plan_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    artifact_path = requested_output
            return {
                "status": "prepared",
                "model_id": self.model_id,
                "artifact_kind": "runtime_profile_plan",
                "artifact_path": str(artifact_path),
                "run_dir": str(run_dir),
                "plan_path": str(artifact_path),
                "command": command,
                "runtime": "worldfoundry.runtime_profile.plan",
                "backend_quality": "plan",
                "profile": self.profile.to_dict(),
            }
        return {
            "status": "blocked",
            "model_id": self.model_id,
            "artifact_kind": "runtime_profile_plan",
            "artifact_path": str(plan_path),
            "run_dir": str(run_dir),
            "plan_path": str(plan_path),
            "command": command,
            "runtime": "worldfoundry.runtime_profile.vendor_blocked",
            "backend_quality": "vendor_blocked",
            "blocked_reason": "official runtime not yet vendored into WorldFoundry",
            "profile": self.profile.to_dict(),
        }


__all__ = [
    "BaseSynthesis",
    "RuntimeProfileSynthesis",
]
