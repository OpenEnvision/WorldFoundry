from __future__ import annotations

import inspect
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core.io import file_sha256, load_serialized, resolve_data_path
from worldfoundry.runtime.assets import expand_worldfoundry_path


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _expand_value(value: Any) -> Any:
    if isinstance(value, str) and ("$WORLDFOUNDRY_" in value or "${WORLDFOUNDRY_" in value):
        return str(expand_worldfoundry_path(value))
    if isinstance(value, str) and value.startswith("models/"):
        return str(resolve_data_path(*Path(value).parts))
    if isinstance(value, Mapping):
        return {str(key): _expand_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_value(item) for item in value]
    return value


def _existing_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    candidates = (path,) if path.is_absolute() else (Path.cwd() / path, _REPO_ROOT / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _first_existing(items: Any) -> Path | None:
    if isinstance(items, (str, Path)):
        return _existing_path(str(items))
    if not isinstance(items, list):
        return None
    for item in items:
        found = _existing_path(item)
        if found is not None:
            return found
    return None


def _missing_required_paths(
    items: Any,
    *,
    repo_root: Path | None,
    checkpoint_path: Path | None,
) -> list[str]:
    if not isinstance(items, list):
        return []

    variables = {
        "repo_root": str(repo_root or ""),
        "checkpoint_path": str(checkpoint_path or ""),
    }
    missing: list[str] = []
    for item in items:
        label = ""
        base = "checkpoint"
        raw_path: Any = item
        if isinstance(item, Mapping):
            raw_path = item.get("path")
            label = str(item.get("id") or item.get("label") or "")
            base = str(item.get("base") or base)
            hint = str(item.get("hint") or "")
        else:
            hint = ""
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        rendered = str(_expand_value(raw_path)).format(**variables)
        candidate = Path(rendered).expanduser()
        if not candidate.is_absolute():
            if base == "repo":
                candidate = (repo_root / candidate) if repo_root else candidate
            elif base == "cwd":
                candidate = candidate
            else:
                candidate = (checkpoint_path / candidate) if checkpoint_path else candidate
        if not candidate.exists():
            prefix = f"{label}: " if label else ""
            suffix = f"; {hint}" if hint else ""
            missing.append(f"{prefix}{rendered} -> {candidate}{suffix}")
    return missing


def _first_complete_checkpoint(
    items: Any,
    *,
    required_paths: Any,
    repo_root: Path | None,
) -> Path | None:
    """Prefer a candidate that satisfies the runtime's required asset layout."""

    sources = [items] if isinstance(items, (str, Path)) else items
    if not isinstance(sources, list):
        return None
    first_existing: Path | None = None
    for item in sources:
        candidate = _existing_path(str(item))
        if candidate is None:
            continue
        if first_existing is None:
            first_existing = candidate
        if not _missing_required_paths(
            required_paths,
            repo_root=repo_root,
            checkpoint_path=candidate,
        ):
            return candidate
    return first_existing


def _format_command(items: list[str], variables: Mapping[str, Any]) -> list[str]:
    rendered: list[str] = []
    format_variables = {
        key: " ".join(str(part) for part in value) if isinstance(value, (list, tuple)) else str(value)
        for key, value in variables.items()
    }
    for item in items:
        placeholder = str(item).strip()
        if placeholder == "{pyramid_num_inference_steps_list}" and isinstance(
            variables.get("pyramid_num_inference_steps_list"), (list, tuple)
        ):
            rendered.extend(str(part) for part in variables["pyramid_num_inference_steps_list"])
            continue
        if str(item).strip() == "{torchrun}" and isinstance(variables.get("torchrun"), (list, tuple)):
            python = str(variables.get("python") or "")
            if python and (not rendered or rendered[-1] != python):
                rendered.append(python)
            rendered.extend(str(part) for part in variables["torchrun"])
            continue
        rendered.append(str(item).format(**format_variables))
    return rendered


def _absolute_cli_path(value: Any) -> Any:
    if isinstance(value, Path):
        return value.expanduser().resolve()
    if not isinstance(value, str) or not value.strip():
        return value
    stripped = value.strip()
    if "://" in stripped:
        return value
    return Path(stripped).expanduser().resolve()


def _resolve_flat_wan_checkpoint_alias(value: Any) -> Any:
    """Resolve the two local mirror names used by official Wan releases."""

    if not isinstance(value, (str, Path)):
        return value
    path = Path(value).expanduser()
    if path.exists() or not path.name.startswith("Wan2."):
        return path.resolve()
    prefixed = path.parent / f"Wan-AI--{path.name}"
    return prefixed.resolve() if prefixed.exists() else path.resolve()


def _normalize_cli_path_variables(variables: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(variables)
    for key, value in tuple(normalized.items()):
        if key in {
            "repo_root",
            "checkpoint_path",
            "checkpoint_parent",
            "output_path",
            "output_dir",
            "image_path",
            "video_path",
            "eval_dir",
            "vae",
            "text_encoder",
            "wan_model_root",
            "wan_root",
        }:
            normalized[key] = _absolute_cli_path(value)
    for key in ("wan_model_root", "wan_root"):
        if key in normalized:
            normalized[key] = _resolve_flat_wan_checkpoint_alias(normalized[key])
    return normalized


def _with_cli_aliases(defaults: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    values = {str(key): value for key, value in extra.items()}
    alias_groups = (
        ("num_sampling_steps", "num_inference_steps", "sampling_steps", "steps", "infer_steps", "num_steps"),
        ("num_inference_steps", "num_sampling_steps", "sampling_steps", "steps", "infer_steps", "num_steps"),
        ("num_steps", "num_inference_steps", "sampling_steps", "steps", "infer_steps", "num_sampling_steps"),
        ("infer_steps", "num_inference_steps", "sampling_steps", "steps", "num_steps", "num_sampling_steps"),
        ("sample_steps", "num_inference_steps", "sampling_steps", "steps", "infer_steps", "num_steps"),
        ("video_length", "num_frames", "frames", "max_frames"),
    )
    for canonical, *aliases in alias_groups:
        if canonical not in defaults or canonical in values:
            continue
        for alias in aliases:
            if alias in values:
                values[canonical] = values[alias]
                break
    return values


def _torchrun_executable() -> tuple[str, str]:
    return ("-m", "torch.distributed.run")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def _normalize_visible_devices(value: Any) -> str:
    if value in {None, ""}:
        return ""
    if isinstance(value, (list, tuple)):
        return ",".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


@dataclass(frozen=True)
class RuntimeRequirementReport:
    missing: tuple[str, ...]
    repo_root: Path | None
    checkpoint_path: Path | None

    @property
    def ready(self) -> bool:
        return not self.missing


class OfficialVideoRuntime:
    """Data-backed runtime adapter for official video-family inference paths.

    The config decides whether a model uses an official CLI checkout, a diffusers
    pipeline, a Hugging Face transformers model, or an API endpoint. This class
    performs real execution for those backends and returns a structured blocked
    result when required local assets are absent.
    """

    def __init__(self, *, model_id: str, runtime_config_path: str, device: str = "cuda", **overrides: Any) -> None:
        self.model_id = model_id
        self.device = device
        self.runtime_config_path = runtime_config_path
        self.config = self._load_config(runtime_config_path)
        runtime = dict(self.config.get("runtime") or {})
        defaults = runtime.get("defaults")
        default_variables = dict(defaults) if isinstance(defaults, Mapping) else {}
        for key, value in overrides.items():
            if value is None:
                continue
            if key in default_variables:
                default_variables[key] = value
            else:
                runtime[key] = value
        if isinstance(defaults, Mapping):
            runtime["defaults"] = default_variables
        self.runtime = _expand_value(runtime)
        expanded_defaults = self.runtime.get("defaults")
        if isinstance(expanded_defaults, dict):
            for key in tuple(expanded_defaults):
                if not key.endswith("_candidates") or not isinstance(expanded_defaults[key], list):
                    continue
                variable = key.removesuffix("_candidates")
                candidates = expanded_defaults.pop(key)
                if overrides.get(variable) is None:
                    resolved = _first_existing(candidates)
                    if resolved is not None:
                        expanded_defaults[variable] = str(resolved)
        self._diffusers_pipeline: Any | None = None
        self._diffusers_pipeline_key: tuple[str, ...] | None = None

    @staticmethod
    def _load_config(path: str) -> dict[str, Any]:
        payload = load_serialized(resolve_data_path(*Path(path).parts))
        if not isinstance(payload, Mapping):
            raise ValueError(f"Official video runtime config must be a mapping: {path}")
        return dict(payload)

    @classmethod
    def from_model_id(cls, model_id: str, *, device: str = "cuda", **overrides: Any) -> "OfficialVideoRuntime":
        return cls(
            model_id=model_id,
            runtime_config_path=f"models/runtime/configs/video_official/{model_id}.yaml",
            device=device,
            **overrides,
        )

    def _requirement_report(self, *, extra: Mapping[str, Any] | None = None) -> RuntimeRequirementReport:
        missing: list[str] = []
        repo_sources = self.runtime.get("repo_root") or self.runtime.get("repo_root_candidates")
        checkpoint_sources = self.runtime.get("checkpoint_path") or self.runtime.get("checkpoint_candidates")
        repo_root = _first_existing(repo_sources)
        checkpoint_path = _first_complete_checkpoint(
            checkpoint_sources,
            required_paths=self.runtime.get("required_paths"),
            repo_root=repo_root,
        )

        kind = str(self.runtime.get("kind") or "")
        if kind == "official_cli" and repo_root is None:
            missing.append(
                "official repo checkout not found; checked "
                f"{repo_sources}"
            )
        checkpoint_optional = bool(self.runtime.get("checkpoint_optional", False))
        if (
            kind in {"official_cli", "diffusers_pipeline", "transformers_generation"}
            and checkpoint_path is None
            and not checkpoint_optional
        ):
            missing.append(
                "checkpoint/assets not found; checked "
                f"{checkpoint_sources}"
            )
        for path in _missing_required_paths(
            self.runtime.get("required_paths"),
            repo_root=repo_root,
            checkpoint_path=checkpoint_path,
        ):
            missing.append(f"required runtime path not found: {path}")
        defaults = self.runtime.get("defaults")
        if isinstance(defaults, Mapping):
            effective_defaults = {**defaults, **_with_cli_aliases(defaults, extra or {})}
            for key in self.runtime.get("required_default_paths") or ():
                value = effective_defaults.get(key)
                candidate = (
                    _resolve_flat_wan_checkpoint_alias(value)
                    if key in {"wan_model_root", "wan_root"} and value
                    else _existing_path(value)
                )
                if candidate is None or not Path(candidate).exists():
                    missing.append(f"required default path not found: {key}={value}")
        if kind == "api_endpoint":
            env_name = str(self.runtime.get("api_key_env") or "")
            if env_name and not os.environ.get(env_name):
                missing.append(f"API key environment variable is not set: {env_name}")
        return RuntimeRequirementReport(tuple(missing), repo_root, checkpoint_path)

    def prepare(self) -> None:
        """Eagerly load reusable in-process backends during pipeline loading.

        Workspace serializes pipeline construction but executes different GPU
        jobs concurrently. Loading Diffusers lazily from ``generate`` both
        defeated the Workspace pipeline cache and allowed its lazy module
        proxy to race with unrelated Transformers/Diffusers imports.
        """

        if str(self.runtime.get("kind") or "") != "diffusers_pipeline":
            return
        report = self._requirement_report()
        if report.ready:
            self._get_diffusers_pipeline(report.checkpoint_path)

    def _get_diffusers_pipeline(
        self,
        checkpoint_path: Path | None,
        *,
        torch_dtype_name: str | None = None,
    ) -> tuple[Any, Any]:
        import importlib

        import torch

        dtype_name = str(torch_dtype_name or self.runtime.get("torch_dtype") or "float16")
        module_name, _, attr_name = str(
            self.runtime.get("pipeline_target") or "diffusers:DiffusionPipeline"
        ).partition(":")
        key = (
            str(checkpoint_path),
            module_name,
            attr_name or "DiffusionPipeline",
            dtype_name,
            str(self.device),
            str(bool(self.runtime.get("enable_model_cpu_offload", False))),
            str(bool(self.runtime.get("enable_vae_tiling", False))),
        )
        if self._diffusers_pipeline is not None and self._diffusers_pipeline_key == key:
            return self._diffusers_pipeline, getattr(torch, dtype_name)

        module = importlib.import_module(module_name)
        pipeline_cls = getattr(module, attr_name or "DiffusionPipeline")
        torch_dtype = getattr(torch, dtype_name)
        pipe = pipeline_cls.from_pretrained(str(checkpoint_path), torch_dtype=torch_dtype)
        enable_model_cpu_offload = bool(self.runtime.get("enable_model_cpu_offload", False))
        if enable_model_cpu_offload:
            offload = getattr(pipe, "enable_model_cpu_offload", None)
            if not callable(offload):
                raise TypeError(
                    f"{pipeline_cls.__name__} does not support requested model CPU offload"
                )
            offload(device=self.device)
        elif hasattr(pipe, "to"):
            pipe = pipe.to(self.device)
        if bool(self.runtime.get("enable_vae_tiling", False)):
            vae = getattr(pipe, "vae", None)
            enable_tiling = getattr(vae, "enable_tiling", None)
            if not callable(enable_tiling):
                raise TypeError(
                    f"{pipeline_cls.__name__} does not support requested VAE tiling"
                )
            enable_tiling()
        self._diffusers_pipeline = pipe
        self._diffusers_pipeline_key = key
        return pipe, torch_dtype

    def runtime_plan(
        self,
        *,
        output_path: str | Path | None = None,
        prompt: str = "",
        image_path: str | Path | None = None,
        video_path: str | Path | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        report = self._requirement_report(extra=extra)
        defaults = self.runtime.get("defaults")
        default_variables = dict(defaults) if isinstance(defaults, Mapping) else {}
        variables = {
            "python": sys.executable,
            "torchrun": _torchrun_executable(),
            "master_port": _find_free_port(),
            "repo_root": report.repo_root or "",
            "checkpoint_path": report.checkpoint_path or "",
            "checkpoint_parent": report.checkpoint_path.parent if report.checkpoint_path else "",
            "output_path": output_path or "",
            "output_dir": Path(output_path).parent if output_path else "",
            "prompt": prompt,
            "image_path": image_path or "",
            "video_path": video_path or (extra or {}).get("video_path") or "",
            "device": self.device,
            **default_variables,
            **_with_cli_aliases(default_variables, extra or {}),
        }
        variables = _normalize_cli_path_variables(variables)
        command = self.runtime.get("command")
        return {
            "model_id": self.model_id,
            "runtime": self.runtime.get("kind"),
            "backend_quality": self.runtime.get("backend_quality", "official_runtime_bridge"),
            "ready": report.ready,
            "missing": list(report.missing),
            "repo_root": str(report.repo_root) if report.repo_root else None,
            "checkpoint_path": str(report.checkpoint_path) if report.checkpoint_path else None,
            "command": _format_command(command, variables) if isinstance(command, list) else None,
            "notes": list(self.config.get("notes") or ()),
        }

    def _blocked_result(
        self,
        *,
        output_path: Path,
        prompt: str,
        missing: tuple[str, ...],
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path = output_path.with_suffix(output_path.suffix + ".runtime_plan.json")
        plan_path.write_text(
            json.dumps(self.runtime_plan(output_path=output_path, prompt=prompt, extra=extra), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return {
            "status": "blocked",
            "runtime": self.runtime.get("kind"),
            "backend_quality": "blocked_missing_runtime_requirements",
            "blocked_reasons": list(missing),
            "blocked_reason": "; ".join(missing),
            "plan_path": str(plan_path),
            "artifact_path": str(output_path),
            "metadata": {"runtime_config_path": self.runtime_config_path},
        }

    def generate(
        self,
        *,
        prompt: str,
        output_path: str | Path,
        image_path: str | Path | None = None,
        video_path: str | Path | None = None,
        plan_only: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        output = Path(output_path).expanduser().resolve()
        report = self._requirement_report(extra=kwargs)
        if plan_only:
            output.parent.mkdir(parents=True, exist_ok=True)
            plan_path = output.with_suffix(output.suffix + ".runtime_plan.json")
            plan_path.write_text(
                json.dumps(
                    self.runtime_plan(
                        output_path=output,
                        prompt=prompt,
                        image_path=image_path,
                        video_path=video_path,
                        extra=kwargs,
                    ),
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            return {
                "status": "blocked" if not report.ready else "planned",
                "runtime": self.runtime.get("kind"),
                "backend_quality": self.runtime.get("backend_quality", "official_runtime_plan"),
                "blocked_reasons": list(report.missing),
                "blocked_reason": "; ".join(report.missing) if report.missing else None,
                "plan_path": str(plan_path),
                "artifact_path": str(output),
            }
        if not report.ready:
            return self._blocked_result(output_path=output, prompt=prompt, missing=report.missing, extra=kwargs)

        kind = str(self.runtime.get("kind"))
        if kind == "official_cli":
            return self._run_official_cli(
                prompt=prompt,
                output_path=output,
                image_path=image_path,
                video_path=video_path,
                repo_root=report.repo_root,
                checkpoint_path=report.checkpoint_path,
                extra=kwargs,
            )
        if kind == "diffusers_pipeline":
            return self._run_diffusers(
                prompt=prompt,
                output_path=output,
                image_path=image_path,
                checkpoint_path=report.checkpoint_path,
                extra=kwargs,
            )
        if kind == "modelscope_text_to_video":
            return self._run_modelscope_text_to_video(
                prompt=prompt,
                output_path=output,
                checkpoint_path=report.checkpoint_path,
                extra=kwargs,
            )
        if kind == "transformers_generation":
            if self.model_id == "omnivinci":
                return self._run_omnivinci(
                    prompt=prompt,
                    output_path=output,
                    image_path=image_path,
                    video_path=video_path,
                    checkpoint_path=report.checkpoint_path,
                    extra=kwargs,
                )
            if self.model_id == "qwen2.5-omni":
                return self._run_qwen25_omni(
                    prompt=prompt,
                    output_path=output,
                    image_path=image_path,
                    video_path=video_path,
                    checkpoint_path=report.checkpoint_path,
                    extra=kwargs,
                )
            if self.model_id in {"spatial-reasoner", "spatial-ladder"}:
                return self._run_qwen25_vl(
                    prompt=prompt,
                    output_path=output,
                    image_path=image_path,
                    video_path=video_path,
                    checkpoint_path=report.checkpoint_path,
                    extra=kwargs,
                )
            return self._run_transformers_text(
                prompt=prompt,
                output_path=output,
                checkpoint_path=report.checkpoint_path,
                extra=kwargs,
            )
        if kind == "api_endpoint":
            return self._blocked_result(
                output_path=output,
                prompt=prompt,
                missing=("API execution is configured but this adapter only validates API credentials in-tree.",),
            )
        raise ValueError(f"Unsupported official video runtime kind for {self.model_id}: {kind!r}")

    def _success_result(self, output: Path, *, metadata: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "status": "succeeded",
            "runtime": self.runtime.get("kind"),
            "backend_quality": self.runtime.get("backend_quality", "official_runtime_bridge"),
            "artifact_path": str(output),
            "artifact_sha256": file_sha256(output) if output.is_file() else None,
            "metadata": dict(metadata),
        }

    @staticmethod
    def _materialize_cli_artifacts(
        produced: Path,
        output_path: Path,
        *,
        since: float,
    ) -> tuple[Path, tuple[Path, ...]]:
        """Copy fresh direct CLI outputs without changing their media type."""

        supported = {".mp4", ".mov", ".webm", ".gif", ".png", ".jpg", ".jpeg", ".wav", ".flac", ".mp3", ".aac"}
        candidates = [produced]
        candidates.extend(
            path
            for path in produced.parent.iterdir()
            if path.is_file()
            and path != produced
            and path.suffix.lower() in supported
            and path.stat().st_mtime >= since - 1.0
        )
        materialized: list[Path] = []
        primary: Path | None = None
        used_suffixes: set[str] = set()
        for candidate in candidates:
            suffix = candidate.suffix.lower() or output_path.suffix.lower()
            if suffix in used_suffixes:
                continue
            used_suffixes.add(suffix)
            target = output_path if suffix == output_path.suffix.lower() else output_path.with_suffix(suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            if candidate.resolve() != target.resolve():
                shutil.copy2(candidate, target)
            if primary is None:
                primary = target
            materialized.append(target)
        if primary is None:
            raise RuntimeError(f"official CLI produced no materializable artifact: {produced}")
        return primary, tuple(materialized)

    def _run_official_cli(
        self,
        *,
        prompt: str,
        output_path: Path,
        image_path: str | Path | None,
        video_path: str | Path | None,
        repo_root: Path | None,
        checkpoint_path: Path | None,
        extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        command = self.runtime.get("command")
        if not isinstance(command, list) or not command:
            raise ValueError(f"{self.model_id} official_cli runtime requires a command list.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        defaults = self.runtime.get("defaults")
        default_variables = dict(defaults) if isinstance(defaults, Mapping) else {}
        variables = {
            "python": sys.executable,
            "torchrun": _torchrun_executable(),
            "master_port": _find_free_port(),
            "repo_root": repo_root or "",
            "checkpoint_path": checkpoint_path or "",
            "checkpoint_parent": checkpoint_path.parent if checkpoint_path else "",
            "output_path": output_path,
            "output_dir": output_path.parent,
            "prompt": prompt,
            "image_path": image_path or "",
            "video_path": video_path or extra.get("video_path") or "",
            "device": self.device,
            **default_variables,
            **_with_cli_aliases(default_variables, extra),
        }
        variables = _normalize_cli_path_variables(variables)
        rendered = _format_command(command, variables)
        cwd_template = self.runtime.get("cwd")
        cwd = str(repo_root) if repo_root else None
        if isinstance(cwd_template, str) and cwd_template.strip():
            cwd = str(Path(cwd_template.format(**{name: str(item) for name, item in variables.items()})).expanduser())
        env = os.environ.copy()
        src_root = Path(__file__).resolve().parents[3]
        prepend_repo_root = bool(self.runtime.get("prepend_repo_root_to_pythonpath", True))
        env["PYTHONPATH"] = os.pathsep.join(
            item
            for item in (
                str(repo_root) if repo_root and prepend_repo_root else "",
                str(src_root),
                env.get("PYTHONPATH", ""),
            )
            if item
        )
        visible_devices = ""
        for source in (extra, self.runtime):
            if not isinstance(source, Mapping):
                continue
            for key in ("cuda_visible_devices", "visible_devices", "cuda_devices", "gpu_ids"):
                visible_devices = _normalize_visible_devices(source.get(key))
                if visible_devices:
                    break
            if visible_devices:
                break
        if visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = visible_devices
        runtime_env = self.runtime.get("env")
        if isinstance(runtime_env, Mapping):
            env.update(
                {
                    str(key): str(value).format(**{name: str(item) for name, item in variables.items()})
                    for key, value in runtime_env.items()
                }
            )
        started_at = time.time()
        completed = subprocess.run(
            rendered,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            log_path = output_path.with_suffix(output_path.suffix + ".log")
            log_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
            return {
                "status": "failed",
                "runtime": "official_cli",
                "backend_quality": "official_cli_failed",
                "artifact_path": str(output_path),
                "error": f"official CLI exited with code {completed.returncode}; see {log_path}",
                "metadata": {"command": rendered, "log_path": str(log_path)},
            }
        log_path = output_path.with_suffix(output_path.suffix + ".log")
        log_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
        produced = self._resolve_produced_artifact(
            output_path,
            search_dirs=(variables.get("output_dir"), output_path.parent),
            since=started_at,
        )
        if produced is None:
            return {
                "status": "failed",
                "runtime": "official_cli",
                "backend_quality": "official_cli_no_artifact",
                "artifact_path": str(output_path),
                "error": f"official CLI completed but no artifact was found at or near {output_path}",
                "metadata": {"command": rendered, "log_path": str(log_path)},
            }
        primary, artifacts = self._materialize_cli_artifacts(produced, output_path, since=started_at)
        result = self._success_result(primary, metadata={"command": rendered, "log_path": str(log_path)})
        result["artifact_paths"] = [str(path) for path in artifacts]
        return result

    def _resolve_produced_artifact(
        self,
        output_path: Path,
        *,
        search_dirs: tuple[Any, ...] = (),
        since: float | None = None,
    ) -> Path | None:
        def _fresh(path: Path) -> bool:
            if since is None:
                return True
            try:
                return path.stat().st_mtime >= since - 1.0
            except OSError:
                return False

        if output_path.is_file() and _fresh(output_path):
            return output_path
        suffixes = {".mp4", ".mov", ".webm", ".gif", ".png", ".jpg", ".jpeg", ".wav", ".flac", ".mp3", ".aac"}
        roots: list[Path] = []
        for raw_root in (*search_dirs, output_path.parent):
            if raw_root in {None, ""}:
                continue
            root = Path(raw_root).expanduser()
            try:
                root = root.resolve()
            except OSError:
                pass
            if root.is_file():
                root = root.parent
            if root.exists() and root not in roots:
                roots.append(root)

        candidates: list[Path] = []
        for root in roots:
            candidates.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in suffixes and _fresh(path)
            )
        candidates = sorted(
            set(candidates),
            key=lambda path: (path.stat().st_mtime, path.stat().st_size),
            reverse=True,
        )
        preferred_patterns = self.runtime.get("preferred_artifact_patterns")
        if isinstance(preferred_patterns, list):
            for pattern in preferred_patterns:
                matches = [
                    path
                    for path in candidates
                    if isinstance(pattern, str) and (fnmatch(path.name, pattern) or fnmatch(str(path), pattern))
                ]
                if matches:
                    return matches[0]
        if candidates:
            return candidates[0]
        if output_path.is_file():
            return output_path
        stale_candidates: list[Path] = []
        for root in roots:
            stale_candidates.extend(
                path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes
            )
        candidates = sorted(
            set(stale_candidates),
            key=lambda path: (path.stat().st_mtime, path.stat().st_size),
            reverse=True,
        )
        return candidates[0] if candidates else None

    def _run_diffusers(
        self,
        *,
        prompt: str,
        output_path: Path,
        image_path: str | Path | None,
        checkpoint_path: Path | None,
        extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        import imageio
        import torch

        call_kwargs = dict(self.runtime.get("call_kwargs") or {})
        call_kwargs.update(extra)
        requested_frames = None
        frame_multiple = int(self.runtime.get("temporal_frame_multiple", 1))
        if frame_multiple > 1 and call_kwargs.get("num_frames") is not None:
            requested_frames = int(call_kwargs["num_frames"])
            if requested_frames < 1:
                raise ValueError("num_frames must be positive")
            # Causal VAEs decode k * stride + 1 frames. Round up before
            # sampling and trim after decoding instead of silently dropping
            # requested frames (e.g. Mochi otherwise turns 84 into 79).
            call_kwargs["num_frames"] = (
                (requested_frames - 1 + frame_multiple - 1) // frame_multiple
            ) * frame_multiple + 1
        dtype_name = str(extra.get("torch_dtype") or self.runtime.get("torch_dtype") or "float16")
        pipe, _ = self._get_diffusers_pipeline(checkpoint_path, torch_dtype_name=dtype_name)
        # Some Diffusers pipelines condition on ``target_fps`` while others do
        # not accept an FPS argument at all. Preserve the requested playback
        # rate before signature filtering so the encoded artifact still honors
        # the Workspace request.
        output_fps = int(
            call_kwargs.get("fps")
            or call_kwargs.get("target_fps")
            or self.runtime.get("fps")
            or 16
        )
        seed = call_kwargs.pop("seed", None)
        signature = inspect.signature(pipe.__call__)
        parameters = signature.parameters
        accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
        # ``fps`` is the Workspace playback control. Pipelines such as
        # I2VGen-XL call the corresponding generation micro-condition
        # ``target_fps``; keep conditioning and encoded playback in sync.
        if extra.get("fps") is not None and "target_fps" in parameters and "target_fps" not in extra:
            call_kwargs["target_fps"] = int(extra["fps"])
        if seed is not None and (accepts_kwargs or "generator" in parameters):
            generator_device = self.device if str(self.device).startswith("cuda") and torch.cuda.is_available() else "cpu"
            call_kwargs["generator"] = torch.Generator(device=generator_device).manual_seed(int(seed))
        if image_path:
            from PIL import Image

            if accepts_kwargs or "image" in parameters:
                call_kwargs["image"] = Image.open(image_path).convert("RGB")
        if not accepts_kwargs:
            call_kwargs = {
                key: value
                for key, value in call_kwargs.items()
                if key in parameters
            }
        autocast_dtype_name = self.runtime.get("autocast_dtype")
        if autocast_dtype_name and str(self.device).startswith("cuda"):
            autocast_context = torch.autocast(
                device_type="cuda",
                dtype=getattr(torch, str(autocast_dtype_name)),
                cache_enabled=bool(self.runtime.get("autocast_cache_enabled", False)),
            )
        else:
            autocast_context = nullcontext()
        with autocast_context:
            result = pipe(prompt=prompt, **call_kwargs)
        frames = getattr(result, "frames", None) or getattr(result, "videos", None) or result[0]
        if frames and isinstance(frames, list) and frames and isinstance(frames[0], list):
            frames = frames[0]
        if requested_frames is not None:
            if len(frames) < requested_frames:
                raise RuntimeError(
                    f"Video decoder returned {len(frames)} frames; expected at least {requested_frames}"
                )
            frames = frames[:requested_frames]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(output_path), frames, fps=output_fps)
        return self._success_result(output_path, metadata={"pipeline_target": self.runtime.get("pipeline_target")})

    def _run_modelscope_text_to_video(
        self,
        *,
        prompt: str,
        output_path: Path,
        checkpoint_path: Path | None,
        extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        import numpy as np
        import torch
        from modelscope.outputs import OutputKeys
        from modelscope.pipelines import pipeline

        seed = int(extra.get("seed") or self.runtime.get("seed") or 42)
        torch.manual_seed(seed)
        np.random.seed(seed)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        original_torch_load = torch.load

        def _torch_load_legacy_default(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = _torch_load_legacy_default  # type: ignore[assignment]
        try:
            pipe = pipeline("text-to-video-synthesis", str(checkpoint_path))
        finally:
            torch.load = original_torch_load  # type: ignore[assignment]
        from modelscope.pipelines.multi_modal.text_to_video_synthesis_pipeline import tensor2vid
        from worldfoundry.core.io.video import save_video_h264

        def _postprocess_video(inputs: dict[str, Any], **post_params: Any) -> dict[str, str]:
            # Preserve ModelScope's frame conversion and encoding parameters while
            # using the shared writer across torchvision/PyAV versions.
            target = Path(post_params.get("output_video") or output_path)
            save_video_h264(tensor2vid(inputs["video"]), target, fps=8, crf=10)
            return {OutputKeys.OUTPUT_VIDEO: str(target)}

        pipe.postprocess = _postprocess_video
        clip_encoder = getattr(getattr(pipe, "model", None), "clip_encoder", None)
        clip_model = getattr(clip_encoder, "model", None)
        if getattr(getattr(clip_model, "transformer", None), "batch_first", False):
            # ModelScope passes LND to individual residual blocks; modern OpenCLIP
            # blocks expect NLD. Preserve the checkpoint's causal attention mask.
            original_forward = clip_encoder.text_transformer_forward

            def _text_transformer_forward(x: Any, attn_mask: Any = None) -> Any:
                return original_forward(x.transpose(0, 1), attn_mask=attn_mask).transpose(0, 1)

            clip_encoder.text_transformer_forward = _text_transformer_forward
        result = pipe({"text": prompt}, output_video=str(output_path))
        produced = result.get(OutputKeys.OUTPUT_VIDEO) if isinstance(result, Mapping) else None
        produced_path = Path(produced).expanduser() if produced else output_path
        if not produced_path.is_file():
            produced_path = self._resolve_produced_artifact(output_path) or produced_path
        if not produced_path.is_file():
            return {
                "status": "failed",
                "runtime": "modelscope_text_to_video",
                "backend_quality": "modelscope_no_artifact",
                "artifact_path": str(output_path),
                "error": f"ModelScope pipeline completed but no artifact was found at {output_path}",
            }
        if produced_path != output_path:
            shutil.copy2(produced_path, output_path)
        return self._success_result(
            output_path,
            metadata={
                "pipeline_target": "modelscope:text-to-video-synthesis",
                "seed": seed,
            },
        )

    def _run_omnivinci(
        self,
        *,
        prompt: str,
        output_path: Path,
        image_path: str | Path | None,
        video_path: str | Path | None,
        checkpoint_path: Path | None,
        extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Follow OmniVinci's released multimodal processor/generation example."""
        if checkpoint_path is None:
            raise ValueError("OmniVinci checkpoint is required")

        video = None
        if video_path is not None:
            video = Path(video_path).expanduser().resolve()
            if not video.is_file():
                raise FileNotFoundError(video)
        requested_frames = int(extra.get("max_video_frames", 8))
        if not 1 <= requested_frames <= 128:
            raise ValueError("OmniVinci max_video_frames must be between 1 and 128")
        load_audio_in_video = extra.get("load_audio_in_video", False)
        if not isinstance(load_audio_in_video, bool):
            raise TypeError("OmniVinci load_audio_in_video must be a bool")

        import torch
        from transformers import AutoModel, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            str(checkpoint_path), trust_remote_code=True, local_files_only=True
        )
        model = AutoModel.from_pretrained(
            str(checkpoint_path),
            trust_remote_code=True,
            torch_dtype=getattr(torch, str(self.runtime.get("torch_dtype") or "float16")),
            device_map=self.runtime.get("device_map") or "auto",
            local_files_only=True,
        )
        model.eval()
        if video is not None:
            # The released example sets both configs before the processor
            # extracts frames. Most QA clips are silent, so audio is opt-in.
            model.config.num_video_frames = requested_frames
            processor.config.num_video_frames = requested_frames
            model.config.load_audio_in_video = load_audio_in_video
            processor.config.load_audio_in_video = load_audio_in_video
        content: list[dict[str, str]] = []
        if image_path:
            image = Path(image_path).expanduser().resolve()
            if not image.is_file():
                raise FileNotFoundError(image)
            content.append({"type": "image", "image": str(image)})
        if video is not None:
            content.append({"type": "video", "video": str(video)})
        content.append({"type": "text", "text": prompt})
        conversation = [{"role": "user", "content": content}]
        templated = processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = processor([templated])
        video_frames_processed = (
            len(inputs.media["video"][0]) if video is not None else 0
        )
        input_ids = inputs.input_ids.to(model.llm_model_embed_tokens.weight.device)
        generation_config = model.default_generation_config
        generation_config.update(max_new_tokens=int(extra.get("max_new_tokens") or 128))
        with torch.inference_mode():
            # OmniVinci's released generate() forwards multimodal embeddings
            # directly to its BF16 Qwen decoder.  Image embeddings can remain
            # FP32 after _embed(), which makes the first decoder matmul fail.
            embeddings, _, attention_mask = model._embed(
                input_ids,
                getattr(inputs, "media", None),
                getattr(inputs, "media_config", None),
                None,
                None,
            )
            decoder_weight = model.llm_model_embed_tokens.weight
            embeddings = embeddings.to(device=decoder_weight.device, dtype=decoder_weight.dtype)
            attention_mask = attention_mask.to(decoder_weight.device)
            output_ids = model.llm.generate(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )
        if output_ids.shape[1] >= input_ids.shape[1] and torch.equal(
            output_ids[:, : input_ids.shape[1]].cpu(), input_ids.cpu()
        ):
            output_ids = output_ids[:, input_ids.shape[1] :]
        answer = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        output = output_path.with_suffix(".json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "prompt": prompt,
                    "text": answer,
                    "image_path": str(image_path or ""),
                    "video_path": str(video or ""),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return self._success_result(
            output,
            metadata={
                "checkpoint_path": str(checkpoint_path),
                "model_class": type(model).__name__,
                "input_kind": (
                    "image_video_text"
                    if video is not None and image_path
                    else "video_text" if video is not None else "image_text" if image_path else "text"
                ),
                "max_video_frames": requested_frames if video is not None else None,
                "video_frames_processed": video_frames_processed if video is not None else None,
                "load_audio_in_video": load_audio_in_video if video is not None else None,
            },
        )

    def _run_qwen25_omni(
        self,
        *,
        prompt: str,
        output_path: Path,
        image_path: str | Path | None,
        video_path: str | Path | None,
        checkpoint_path: Path | None,
        extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Use Qwen2.5-Omni's conditional generation API for visual QA."""
        if video_path and image_path:
            raise ValueError("Qwen2.5-Omni accepts one image or one video per request")
        if checkpoint_path is None:
            raise ValueError("Qwen2.5-Omni checkpoint is required")

        import torch
        from PIL import Image
        from transformers import (
            Qwen2_5OmniConfig,
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniProcessor,
        )

        processor = Qwen2_5OmniProcessor.from_pretrained(str(checkpoint_path))
        config = Qwen2_5OmniConfig.from_pretrained(str(checkpoint_path))
        config.enable_audio_output = False
        # The Omni override always reads spk_dict.pt, even for text-only output.
        # Load the safetensors weights through the Transformers base loader; the
        # speaker pickle and audio-output modules are unnecessary for image QA.
        model = super(
            Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniForConditionalGeneration
        ).from_pretrained(
            str(checkpoint_path),
            config=config,
            torch_dtype=getattr(torch, str(self.runtime.get("torch_dtype") or "bfloat16")),
            device_map=self.runtime.get("device_map") or "auto",
            use_safetensors=True,
            local_files_only=True,
        )
        model.eval()

        content: list[dict[str, str]] = []
        image = None
        if image_path:
            image = Image.open(image_path).convert("RGB")
            content.append({"type": "image", "image": str(image_path)})
        video = video_metadata = None
        if video_path:
            video, video_metadata = self._sample_qwen_video(
                video_path, int(extra.get("max_video_frames") or 8)
            )
            content.append({"type": "video", "video": str(video_path)})
        content.append({"type": "text", "text": prompt})
        conversation = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        processor_kwargs: dict[str, Any] = {
            "text": text,
            "images": [image] if image is not None else None,
            "videos": [video] if video is not None else None,
            "return_tensors": "pt",
            "padding": True,
        }
        if video_metadata is not None:
            processor_kwargs["fps"] = video_metadata["sampled_fps"]
        inputs = processor(**processor_kwargs)
        if video_metadata is not None:
            if "pixel_values_videos" not in inputs or "video_grid_thw" not in inputs:
                raise RuntimeError("Qwen2.5-Omni processor omitted required video tensors")
            video_metadata["pixel_values_videos_shape"] = list(inputs["pixel_values_videos"].shape)
            video_metadata["video_grid_thw"] = inputs["video_grid_thw"].tolist()
            video_token_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)
            video_metadata["video_token_count"] = int(
                (inputs["input_ids"] == video_token_id).sum().item()
            )
            if video_metadata["video_token_count"] == 0:
                raise RuntimeError("Qwen2.5-Omni processor produced no video tokens")
        inputs = inputs.to(model.device).to(model.dtype)
        visual_forward_calls: list[dict[str, Any]] = []
        visual_hook = None
        if video_metadata is not None:
            def _capture_visual_forward(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
                pixels = args[0] if args else kwargs.get("hidden_states")
                grid = kwargs.get("grid_thw")
                visual_forward_calls.append({
                    "pixel_shape": list(pixels.shape) if isinstance(pixels, torch.Tensor) else None,
                    "grid_thw": grid.tolist() if isinstance(grid, torch.Tensor) else None,
                })

            visual_hook = model.thinker.visual.register_forward_pre_hook(_capture_visual_forward, with_kwargs=True)
        try:
            with torch.inference_mode():
                output_ids = model.thinker.generate(
                    **inputs,
                    max_new_tokens=int(extra.get("max_new_tokens") or 128),
                )
        finally:
            if visual_hook is not None:
                visual_hook.remove()
        if video_metadata is not None:
            video_metadata["visual_forward_calls"] = visual_forward_calls
            if not visual_forward_calls or not any(
                call["pixel_shape"] == video_metadata["pixel_values_videos_shape"]
                and call["grid_thw"] == video_metadata["video_grid_thw"]
                for call in visual_forward_calls
            ):
                raise RuntimeError("Qwen2.5-Omni video tensor did not reach the visual encoder")
        prompt_ids = inputs["input_ids"]
        if output_ids.shape[1] >= prompt_ids.shape[1] and torch.equal(
            output_ids[:, : prompt_ids.shape[1]], prompt_ids
        ):
            output_ids = output_ids[:, prompt_ids.shape[1] :]
        answer = processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        output = output_path.with_suffix(".json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "prompt": prompt,
                    "text": answer,
                    "image_path": str(image_path or ""),
                    "video_input": video_metadata,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return self._success_result(
            output,
            metadata={
                "checkpoint_path": str(checkpoint_path),
                "model_class": "Qwen2_5OmniForConditionalGeneration.thinker",
                "input_kind": "video_text" if video is not None else "image_text" if image is not None else "text",
            },
        )

    @staticmethod
    def _sample_qwen_video(video_path: str | Path, max_frames: int) -> tuple[Any, dict[str, Any]]:
        """Decode a bounded, evenly spaced set of RGB frames for Qwen2.5-VL."""
        import cv2
        import hashlib
        import numpy as np

        path = Path(video_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if not 4 <= max_frames <= 128:
            raise ValueError("Qwen2.5 video input requires 4 <= max_video_frames <= 128")

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"Cannot decode video input: {path}")
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            source_fps = float(capture.get(cv2.CAP_PROP_FPS))
            if frame_count > 0:
                indices = np.linspace(0, frame_count - 1, min(max_frames, frame_count), dtype=int)
                indices = sorted({int(index) for index in indices})
            else:
                indices = list(range(max_frames))
            frames = []
            for index in indices:
                if frame_count > 0:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Failed to decode video frame {index}: {path}")
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()

        if len(frames) < 4:
            raise ValueError(f"Qwen2.5-VL video input needs at least four frames: {path}")
        sampled_fps = source_fps if source_fps > 0 else 2.0
        if source_fps > 0 and len(indices) > 1 and indices[-1] > indices[0]:
            sampled_fps = source_fps * (len(indices) - 1) / (indices[-1] - indices[0])
        return np.stack(frames), {
            "video_path": str(path),
            "source_sha256": file_sha256(path),
            "source_frame_count": frame_count,
            "source_fps": source_fps,
            "sampled_frame_indices": indices,
            "sampled_fps": sampled_fps,
            "sampled_frame_shape": list(frames[0].shape),
            "sampled_rgb_frame_sha256": [hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames],
        }

    def _run_qwen25_vl(
        self,
        *,
        prompt: str,
        output_path: Path,
        image_path: str | Path | None,
        video_path: str | Path | None,
        checkpoint_path: Path | None,
        extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Run a released Qwen2.5-VL checkpoint with its visual inputs."""
        if video_path and image_path:
            raise ValueError(f"{self.model_id} accepts one image or one video per request")
        if checkpoint_path is None:
            raise ValueError(f"{self.model_id} checkpoint is required")

        import torch
        from PIL import Image
        from transformers import (
            AutoProcessor,
            AutoTokenizer,
            Qwen2VLImageProcessor,
            Qwen2VLVideoProcessor,
            Qwen2_5_VLForConditionalGeneration,
            Qwen2_5_VLProcessor,
        )

        preprocessor_config = json.loads((checkpoint_path / "preprocessor_config.json").read_text(encoding="utf-8"))
        if preprocessor_config.get("image_processor_type") == "Qwen2_5_VLImageProcessor":
            # The released SpatialReasoner names a processor class absent from
            # current Transformers. Qwen2.5-VL uses Qwen2VL image preprocessing.
            image_processor = Qwen2VLImageProcessor.from_pretrained(
                str(checkpoint_path), local_files_only=True
            )
            tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path), local_files_only=True)
            chat_config = json.loads((checkpoint_path / "chat_template.json").read_text(encoding="utf-8"))
            processor = Qwen2_5_VLProcessor(
                image_processor=image_processor,
                tokenizer=tokenizer,
                video_processor=Qwen2VLVideoProcessor(),
                chat_template=chat_config["chat_template"],
            )
        else:
            processor = AutoProcessor.from_pretrained(str(checkpoint_path), local_files_only=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(checkpoint_path),
            torch_dtype=getattr(torch, str(self.runtime.get("torch_dtype") or "bfloat16")),
            device_map=self.runtime.get("device_map") or "auto",
            use_safetensors=True,
            local_files_only=True,
        )
        model.eval()
        image = Image.open(image_path).convert("RGB") if image_path else None
        video = video_metadata = None
        if video_path:
            video, video_metadata = self._sample_qwen_video(
                video_path, int(extra.get("max_video_frames") or 8)
            )
        content: list[dict[str, str]] = []
        if image is not None:
            content.append({"type": "image", "image": str(image_path)})
        if video is not None:
            content.append({"type": "video", "video": str(video_path)})
        content.append({"type": "text", "text": prompt})
        conversation = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        processor_kwargs: dict[str, Any] = {
            "text": [text],
            "images": [image] if image is not None else None,
            "videos": [video] if video is not None else None,
            "padding": True,
            "return_tensors": "pt",
        }
        if video_metadata is not None:
            processor_kwargs["fps"] = video_metadata["sampled_fps"]
        inputs = processor(**processor_kwargs)
        if video is not None:
            if "pixel_values_videos" not in inputs or "video_grid_thw" not in inputs:
                raise RuntimeError("Qwen2.5-VL processor omitted required video tensors")
            video_metadata["video_grid_thw"] = inputs["video_grid_thw"].tolist()
            video_metadata["pixel_values_videos_shape"] = list(inputs["pixel_values_videos"].shape)
            video_metadata["video_token_count"] = int(
                (inputs["input_ids"] == processor.video_token_id).sum().item()
            )
            if video_metadata["video_token_count"] == 0:
                raise RuntimeError("Qwen2.5-VL processor produced no video tokens")
        inputs = inputs.to(model.device)
        visual_forward_calls: list[dict[str, Any]] = []
        visual_hook = None
        if video is not None:
            def _capture_visual_forward(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
                pixels = args[0] if args else kwargs.get("hidden_states")
                grid = kwargs.get("grid_thw")
                visual_forward_calls.append({
                    "pixel_shape": list(pixels.shape) if isinstance(pixels, torch.Tensor) else None,
                    "grid_thw": grid.tolist() if isinstance(grid, torch.Tensor) else None,
                })

            visual_hook = model.visual.register_forward_pre_hook(_capture_visual_forward, with_kwargs=True)
        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs, max_new_tokens=int(extra.get("max_new_tokens") or 128), do_sample=False
                )
        finally:
            if visual_hook is not None:
                visual_hook.remove()
        if video_metadata is not None:
            video_metadata["visual_forward_calls"] = visual_forward_calls
            if not visual_forward_calls or not any(
                call["pixel_shape"] == video_metadata["pixel_values_videos_shape"]
                and call["grid_thw"] == video_metadata["video_grid_thw"]
                for call in visual_forward_calls
            ):
                raise RuntimeError("Qwen2.5-VL video tensor did not reach the visual encoder")
        answer_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        answer = processor.batch_decode(
            answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        output = output_path.with_suffix(".json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "prompt": prompt,
                    "text": answer,
                    "image_path": str(image_path or ""),
                    "video_input": video_metadata,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return self._success_result(
            output,
            metadata={
                "checkpoint_path": str(checkpoint_path),
                "model_class": "Qwen2_5_VLForConditionalGeneration",
                "input_kind": "video_text" if video is not None else "image_text" if image is not None else "text",
            },
        )

    def _run_transformers_text(
        self,
        *,
        prompt: str,
        output_path: Path,
        checkpoint_path: Path | None,
        extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        trust_remote_code = bool(self.runtime.get("trust_remote_code", True))
        processor = None
        try:
            processor = AutoProcessor.from_pretrained(str(checkpoint_path), trust_remote_code=trust_remote_code)
        except Exception:
            processor = AutoTokenizer.from_pretrained(str(checkpoint_path), trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_path),
            trust_remote_code=trust_remote_code,
            torch_dtype=getattr(torch, str(self.runtime.get("torch_dtype") or "float16")),
            device_map=self.runtime.get("device_map") or "auto",
        )
        inputs = processor(prompt, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=int(extra.get("max_new_tokens") or 128))
        text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        output_path = output_path.with_suffix(".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"prompt": prompt, "text": text}, ensure_ascii=False, indent=2), encoding="utf-8")
        return self._success_result(output_path, metadata={"checkpoint_path": str(checkpoint_path)})


__all__ = ["OfficialVideoRuntime"]
