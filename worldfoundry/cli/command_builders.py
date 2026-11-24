# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Shell command builders for inference, model x benchmark runs, suites, and Studio."""

from __future__ import annotations

import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

from worldfoundry.cli.inference_variants import (
    INFER_MODEL_VARIANTS,
    _script_infer_family_id,
    _studio_runtime_target_for_script_variant,
    is_studio_infer_model_id,
    resolve_infer_model_variant,
)

from .utils import append_optional_arg


def build_model_benchmark_command(
    *,
    model_id: str,
    benchmark_id: str,
    output_dir: str | Path,
    mode: str = "official-run",
    model_manifest_dir: str | Path | None = None,
    benchmark_manifest_dir: str | Path | None = None,
    model_variant: str | None = None,
    requests_path: str | Path | None = None,
    task_name: str | None = None,
    generated_artifact_dir: str | Path | None = None,
    output_artifact: str | None = None,
    metrics: Sequence[str] = (),
    json_output: bool = False,
) -> tuple[str, ...]:
    """Build a ``worldfoundry run`` shell command tuple for evaluation.

    Args:
        model_id: Model family identifier.
        benchmark_id: Benchmark identifier.
        output_dir: Output directory path.
        mode: Run mode label (default ``"official-run"``).
        model_manifest_dir: Optional model manifest directory override.
        benchmark_manifest_dir: Optional benchmark manifest directory override.
        model_variant: Optional model variant override.
        requests_path: Optional requests JSON path override.
        task_name: Optional task name override.
        generated_artifact_dir: Optional generated artifact directory override.
        output_artifact: Optional output artifact override.
        metrics: Metric identifiers to evaluate.
        json_output: Whether to add ``--json`` flag.

    Returns:
        Shell command tuple for ``asyncio.create_subprocess_exec``.
    """
    if generated_artifact_dir is not None:
        command = [
            *_worldfoundry_cli_prefix(),
            "score",
            "--benchmark",
            str(benchmark_id),
            "--artifacts",
            str(generated_artifact_dir),
            "--output-dir",
            str(output_dir),
        ]
        _append_optional(command, "--mode", mode)
        _append_optional_path(command, "--benchmark-manifest-dir", benchmark_manifest_dir)
        if json_output:
            command.append("--json")
        return tuple(command)

    command = [
        *_worldfoundry_cli_prefix(),
        "run",
        "--model",
        str(model_id),
        "--benchmark",
        str(benchmark_id),
        "--output-dir",
        str(output_dir),
    ]
    _append_optional(command, "--mode", mode)
    _append_optional_path(command, "--model-manifest-dir", model_manifest_dir)
    _append_optional_path(command, "--benchmark-manifest-dir", benchmark_manifest_dir)
    _append_optional(command, "--model-variant", model_variant)
    _append_optional_path(command, "--requests-path", requests_path)
    _append_optional(command, "--task-name", task_name)
    _append_optional_path(command, "--generated-artifact-dir", generated_artifact_dir)
    _append_optional(command, "--output-artifact", output_artifact)
    for metric in metrics:
        command.extend(["--metric", str(metric)])
    if json_output:
        command.append("--json")
    return tuple(command)


def _build_workspace_infer_command(
    *,
    model_id: str,
    variant_id: str,
    output_dir: str | Path,
    ckpt_root: str | Path | None,
    conda_envs_root: str | Path | None,
    gpu: str | None,
    values: Sequence[tuple[str, object | None]],
) -> tuple[str, ...]:
    """Build one validated ``workspace_job infer`` command."""
    command = [
        sys.executable,
        "-m",
        "worldfoundry.studio.workspace_job",
        "infer",
        "--model-id",
        model_id,
        "--variant-id",
        variant_id,
        "--output-dir",
        str(output_dir),
    ]
    for flag, value in values:
        _append_optional(command, flag, value)

    env_values: list[str] = []
    if ckpt_root is not None:
        env_values.append(f"WORLDFOUNDRY_CKPT_DIR={Path(ckpt_root).expanduser()}")
    if conda_envs_root is not None:
        env_root = Path(conda_envs_root).expanduser()
        env_values.extend(
            (
                f"WORLDFOUNDRY_CONDA_ENVS_ROOT={env_root}",
                f"WORLDFOUNDRY_CONDA_ENV_ROOT={env_root}",
            )
        )
    if gpu:
        env_values.append(f"CUDA_VISIBLE_DEVICES={gpu}")
    if env_values:
        command = ["env", *env_values, *command]
    return tuple(command)


def build_model_infer_command(
    *,
    model_id: str,
    output_dir: str | Path,
    ckpt_type: str | None = None,
    ckpt_root: str | Path | None = None,
    ckpt_path: str | Path | None = None,
    input_path: str | Path | None = None,
    input_dir: str | Path | None = None,
    video_path: str | Path | None = None,
    trajectory_file: str | Path | None = None,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    task: str | None = None,
    mode: str | None = None,
    resize_mode: str | None = None,
    size: str | None = None,
    frames: str | int | None = None,
    steps: str | int | None = None,
    frames_per_generation: str | int | None = None,
    guidance_scale: str | int | float | None = None,
    seed: str | int | None = None,
    fps: str | int | None = None,
    dtype: str | None = None,
    max_sequence_length: str | int | None = None,
    cam_type: str | int | None = None,
    interactions: str | None = None,
    output_formats: str | None = None,
    trajectory: str | None = None,
    angle: str | int | float | None = None,
    distance: str | int | float | None = None,
    orbit_radius: str | int | float | None = None,
    zoom_ratio: str | int | float | None = None,
    alpha_threshold: str | int | float | None = None,
    static_scene: object | None = None,
    low_vram: object | None = None,
    disable_lora: object | None = None,
    vis_rendering: object | None = None,
    offload_t5: object | None = None,
    offload_transformer_during_vae: object | None = None,
    offload_vae: object | None = None,
    output_path: str | Path | None = None,
    conda_envs_root: str | Path | None = None,
    gpu: str | None = None,
    call_json: str | None = None,
    load_json: str | None = None,
) -> tuple[str, ...]:
    """Build a shell command tuple for model inference.

    Resolves legacy model aliases to a concrete Studio inference contract and
    applies variant-specific defaults (e.g. Astra camera type and Neoverse
    trajectory mode).

    Args:
        model_id: Model family identifier.
        output_dir: Output directory path.
        ckpt_type: Checkpoint variant / function selector.
        ckpt_root: Checkpoint root directory override.
        ckpt_path: Exact checkpoint path or HuggingFace repo override.
        input_path: Input image/video path override.
        input_dir: Input directory override (FlashWorld JSON, etc.).
        video_path: Input video path override.
        trajectory_file: Custom camera trajectory JSON override.
        prompt: Text prompt override.
        negative_prompt: Negative prompt override.
        task: Task mode override.
        mode: Generation mode override.
        resize_mode: Resize mode override.
        size: Resolution override (e.g. ``"832*480"``).
        frames: Frame count override.
        steps: Sampling step override.
        frames_per_generation: Astra frame chunk override.
        guidance_scale: CFG / guidance scale override.
        seed: Random seed override.
        fps: Output FPS override.
        dtype: Runtime dtype override.
        max_sequence_length: Text encoder length override.
        cam_type: Astra camera preset override.
        interactions: Interaction / action list override.
        output_formats: FlashWorld output formats override.
        trajectory: Camera trajectory preset override.
        angle: Camera angle override.
        distance: Camera distance override.
        orbit_radius: Camera orbit radius override.
        zoom_ratio: Camera zoom ratio override.
        alpha_threshold: Alpha threshold override.
        static_scene: Static scene flag.
        low_vram: Low-VRAM mode flag.
        disable_lora: Disable LoRA flag.
        vis_rendering: Rendering visualization flag.
        offload_t5: T5 offload flag.
        offload_transformer_during_vae: Transformer offload during VAE flag.
        offload_vae: VAE offload flag.
        output_path: Exact output artifact path override.
        conda_envs_root: Conda envs root directory override.
        gpu: CUDA_VISIBLE_DEVICES override.

    Returns:
        Shell command tuple for ``asyncio.create_subprocess_exec``.

    Raises:
        ValueError: When the model has no registered infer script or variant group.
    """
    for label, value in (("call_json", call_json), ("load_json", load_json)):
        if value:
            import json

            if not isinstance(json.loads(value), dict):
                raise ValueError(f"{label} must be a JSON object")
    resolved_variant = resolve_infer_model_variant(model_id, ckpt_type)
    script_family = _script_infer_family_id(model_id)
    script_infer = script_family in INFER_MODEL_VARIANTS
    if not script_infer and not is_studio_infer_model_id(model_id):
        raise ValueError(f"no inference runtime is registered for {model_id!r}")
    # NOTE: Apply variant-specific defaults before building the command
    if resolved_variant == "astra":
        frames = frames or 24
        frames_per_generation = frames_per_generation or 8
        fps = fps or 20
        cam_type = cam_type or 4
    if resolved_variant == "flash-world":
        fps = fps or 15
        output_formats = output_formats or "video,spz,ply"
        if input_dir is None:
            frames = frames or 24
    if resolved_variant == "neoverse":
        # NOTE: Neoverse defaults — trajectory ``tilt_up``, mode ``relative``, 81 frames
        if not trajectory and not trajectory_file and not interactions:
            trajectory = "tilt_up"
        mode = mode or "relative"
        resize_mode = resize_mode or "center_crop"
        size = size or "560*336"
        frames = frames or 81
        disable_lora_enabled = _truthy(disable_lora)
        steps = steps or (50 if disable_lora_enabled else 4)
        guidance_scale = guidance_scale or (5.0 if disable_lora_enabled else 1.0)
        seed = seed or 42
        fps = fps or 16
        alpha_threshold = alpha_threshold or 1.0
    if resolved_variant == "recammaster" and not trajectory:
        trajectory = "100,100,0,0,30"
    if script_infer:
        command_model, command_variant, _label = _studio_runtime_target_for_script_variant(
            model_id,
            resolved_variant,
        )
    else:
        command_model, command_variant = model_id, resolved_variant
    return _build_workspace_infer_command(
        model_id=command_model,
        variant_id=command_variant,
        output_dir=output_dir,
        ckpt_root=ckpt_root,
        conda_envs_root=conda_envs_root,
        gpu=gpu,
        values=(
            ("--model-ref", ckpt_path),
            ("--input-path", input_path),
            ("--input-dir", input_dir),
            ("--video-path", video_path),
            ("--trajectory-file", trajectory_file),
            ("--prompt", prompt),
            ("--negative-prompt", negative_prompt),
            ("--interactions", interactions),
            ("--task", task),
            ("--mode", mode),
            ("--resize-mode", resize_mode),
            ("--size", size),
            ("--frames", frames),
            ("--steps", steps),
            ("--frames-per-generation", frames_per_generation),
            ("--guidance-scale", guidance_scale),
            ("--seed", seed),
            ("--fps", fps),
            ("--dtype", dtype),
            ("--max-sequence-length", max_sequence_length),
            ("--cam-type", cam_type),
            ("--output-formats", output_formats),
            ("--trajectory", trajectory),
            ("--angle", angle),
            ("--distance", distance),
            ("--orbit-radius", orbit_radius),
            ("--zoom-ratio", zoom_ratio),
            ("--alpha-threshold", alpha_threshold),
            ("--static-scene", static_scene),
            ("--low-vram", low_vram),
            ("--disable-lora", disable_lora),
            ("--vis-rendering", vis_rendering),
            ("--offload-t5", offload_t5),
            ("--offload-transformer-during-vae", offload_transformer_during_vae),
            ("--offload-vae", offload_vae),
            ("--output-path", output_path),
            ("--call-json", call_json),
            ("--load-json", load_json),
        ),
    )


def build_studio_command(*, host: str = "127.0.0.1", port: int = 7860) -> tuple[str, ...]:
    """Build a shell command tuple that launches the local Studio web UI."""
    return (
        sys.executable,
        "-m",
        "worldfoundry.studio.cli",
        "--host",
        host,
        "--port",
        str(port),
    )


def build_suite_command(
    *,
    output_dir: str | Path,
    mode: str = "official-run",
    model_ids: Sequence[str] = (),
    benchmark_ids: Sequence[str] = (),
    suite_ids: Sequence[str] = (),
    model_manifest_dir: str | Path | None = None,
    benchmark_manifest_dir: str | Path | None = None,
    plan_only: bool = True,
) -> tuple[str, ...]:
    """Build a ``worldfoundry run`` shell command tuple for a model × benchmark suite.

    Args:
        output_dir: Output directory path for the suite plan / results.
        mode: Run mode label (default ``"official-run"``).
        model_ids: Model identifiers to include in the suite.
        benchmark_ids: Benchmark identifiers to include in the suite.
        suite_ids: Suite identifiers (alternative to explicit model/benchmark lists).
        model_manifest_dir: Optional model manifest directory override.
        benchmark_manifest_dir: Optional benchmark manifest directory override.
        plan_only: Whether to write a plan only (default ``True``).

    Returns:
        Shell command tuple for ``asyncio.create_subprocess_exec``.
    """
    command = [
        *_worldfoundry_cli_prefix(),
        "run",
        "--output-dir",
        str(output_dir),
    ]
    _append_optional(command, "--mode", mode)
    _append_optional_path(command, "--model-manifest-dir", model_manifest_dir)
    _append_optional_path(command, "--benchmark-manifest-dir", benchmark_manifest_dir)
    for model_id in model_ids:
        command.extend(["--model", str(model_id)])
    for benchmark_id in benchmark_ids:
        command.extend(["--benchmark", str(benchmark_id)])
    for suite_id in suite_ids:
        command.extend(["--suite", str(suite_id)])
    if plan_only:
        command.append("--plan-only")
    return tuple(command)


def _worldfoundry_cli_prefix() -> tuple[str, str, str]:
    """Return the ``python -m worldfoundry.cli`` prefix for shell command tuples."""
    return (sys.executable, "-m", "worldfoundry.cli")


def format_shell_command(command: Sequence[str]) -> str:
    """Join a command tuple into a single shell-escaped string using :func:`shlex.join`."""
    return shlex.join([str(item) for item in command])


# Both legacy names bind to the shared helper in ``cli.utils`` (CM-24); the
# path variant existed only for type-hint readability at call sites.
_append_optional = append_optional_arg


_append_optional_path = append_optional_arg


def _truthy(value: object | None) -> bool:
    """Return whether *value* is truthy — treating ``"1"``/``"true"``/``"yes"``/``"on"`` as true."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "on"}


def _append_optional_flag(command: list[str], flag: str, value: object | None) -> None:
    """Append a boolean *flag* to *command* when *value* is truthy per :func:`_truthy`."""
    if _truthy(value):
        command.append(flag)
