"""CLI entry-point and argument parser for the WorldFoundry TUI.

Provides the ``worldfoundry-tui`` command-line interface that can launch the
full Textual TUI app, a standard-library fallback dashboard, or perform
one-off actions like printing a command or writing a suite plan.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .help import WorldFoundryArgumentParser
from .tui_brand import render_fallback_header
from .tui_discovery import (
    INFER_VARIANT_GROUPS,
    INFER_VARIANT_TO_MODEL,
    build_model_benchmark_command,
    build_model_infer_command,
    build_studio_command,
    build_suite_command,
    format_shell_command,
    infer_control_fields,
    infer_variant_options,
    is_infer_model_row,
    load_tui_catalog,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the ``worldfoundry-tui`` argument parser with inference, evaluation, and Studio flags."""
    parser = WorldFoundryArgumentParser(
        prog="worldfoundry-tui",
        description="Launch real WorldFoundry model inference, benchmark evaluation, and Studio commands.",
    )
    parser.add_argument("--model-manifest-dir", type=Path)
    parser.add_argument("--benchmark-manifest-dir", type=Path)
    parser.add_argument("--runtime-profile-dir", type=Path)
    parser.add_argument("--suite", action="append", dest="suite_ids", default=None)
    parser.add_argument("--model-id")
    parser.add_argument("--benchmark-id")
    parser.add_argument("--metric", action="append", default=None, help="Benchmark metric id. Repeatable or comma-separated.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tui"))
    parser.add_argument("--input", type=Path, help="Optional input image/video passed to model inference scripts.")
    parser.add_argument("--input-dir", "--input_dir", dest="input_dir", type=Path, help="FlashWorld JSON input directory/file.")
    parser.add_argument("--video", type=Path, help="Optional input video passed to model inference scripts.")
    parser.add_argument("--trajectory-file", type=Path, help="Optional custom camera trajectory JSON file.")
    parser.add_argument("--prompt", help="Prompt override for text/image/video generation inference.")
    parser.add_argument("--negative-prompt", dest="negative_prompt", help="Negative prompt override when supported.")
    parser.add_argument("--task", help="Task override for inference scripts with task modes.")
    parser.add_argument("--mode", help="Mode override for inference scripts with mode variants.")
    parser.add_argument("--resize-mode", help="Input resize mode override when supported.")
    parser.add_argument("--size", help="Resolution override, e.g. 832*480.")
    parser.add_argument("--frames", help="Frame count override.")
    parser.add_argument("--steps", help="Sampling/inference step override.")
    parser.add_argument("--frames-per-generation", "--chunk-frames", dest="frames_per_generation", help="Astra frame chunk size.")
    parser.add_argument("--guidance-scale", "--cfg-scale", dest="guidance_scale", help="Guidance/CFG scale override.")
    parser.add_argument("--seed", help="Seed override.")
    parser.add_argument("--fps", help="Output FPS override.")
    parser.add_argument("--dtype", help="Runtime dtype override when supported.")
    parser.add_argument("--max-sequence-length", dest="max_sequence_length", help="Text encoder max sequence length override.")
    parser.add_argument("--cam-type", "--camera-type", dest="cam_type", help="Astra camera preset 1..7.")
    parser.add_argument("--interactions", "--actions", dest="interactions", help="Comma-separated interaction/action list.")
    parser.add_argument("--output-formats", "--output_formats", dest="output_formats", help="FlashWorld output formats, e.g. video,spz,ply.")
    parser.add_argument("--trajectory", "--camera-trajectory", dest="trajectory", help="Camera trajectory override.")
    parser.add_argument("--angle", help="Camera trajectory angle override when supported.")
    parser.add_argument("--distance", help="Camera trajectory distance override when supported.")
    parser.add_argument("--orbit-radius", dest="orbit_radius", help="Camera orbit radius override when supported.")
    parser.add_argument("--zoom-ratio", dest="zoom_ratio", help="Camera zoom ratio override when supported.")
    parser.add_argument("--alpha-threshold", dest="alpha_threshold", help="Alpha threshold override when supported.")
    parser.add_argument(
        "--static-scene",
        action="store_true",
        default=None,
        help="Treat input as a static scene when supported.",
    )
    parser.add_argument("--low-vram", action="store_true", default=None, help="Enable low-VRAM mode when supported.")
    parser.add_argument(
        "--disable-lora",
        action="store_true",
        default=None,
        help="Disable LoRA acceleration when supported.",
    )
    parser.add_argument(
        "--vis-rendering",
        action="store_true",
        default=None,
        help="Save rendering visualization artifacts when supported.",
    )
    parser.add_argument("--offload-t5", "--offload_t5", dest="offload_t5", action="store_true", default=None)
    parser.add_argument(
        "--offload-transformer-during-vae",
        "--offload_transformer_during_vae",
        dest="offload_transformer_during_vae",
        action="store_true",
        default=None,
    )
    parser.add_argument("--offload-vae", "--offload_vae", dest="offload_vae", action="store_true", default=None)
    parser.add_argument("--output-path", type=Path, help="Exact output artifact path when supported.")
    parser.add_argument("--ckpt-type", "--weight-type", dest="ckpt_type", help="Checkpoint type inside the selected model family.")
    parser.add_argument("--ckpt-root", type=Path, help="Checkpoint root for model inference scripts.")
    parser.add_argument("--ckpt-path", type=Path, help="Exact checkpoint/model path override for the selected infer model.")
    parser.add_argument("--conda-envs-root", type=Path, help="Conda env root for model inference scripts.")
    parser.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES value for model inference scripts.")
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Render the dependency-free terminal dashboard and exit.",
    )
    parser.add_argument(
        "--catalog-json",
        action="store_true",
        help="Print the TUI catalog as JSON and exit without importing Textual.",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the model-benchmark command for the selected IDs and exit.",
    )
    parser.add_argument(
        "--write-suite-plan",
        action="store_true",
        help="Write a model x benchmark suite plan for the selected IDs and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the TUI action based on CLI flags.

    Supports four modes beyond the default Textual TUI:

    * ``--catalog-json`` — print the catalog as JSON and exit.
    * ``--write-suite-plan`` — write a model × benchmark suite plan and exit.
    * ``--print-command`` — print the resolved shell command and exit.
    * ``--fallback`` — render a plain-text dashboard and exit.

    Returns:
        Exit code (0 on success).
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # ── Catalog JSON mode ────────────────────────────────────────────
    if args.catalog_json:
        catalog = load_tui_catalog(
            model_manifest_dir=args.model_manifest_dir,
            benchmark_manifest_dir=args.benchmark_manifest_dir,
            runtime_profile_dir=args.runtime_profile_dir,
            conda_envs_root=args.conda_envs_root,
        )
        _safe_print(json.dumps(catalog.to_dict(), indent=2, ensure_ascii=False))
        return 0

    # ── Suite-plan mode ──────────────────────────────────────────────
    if args.write_suite_plan:
        catalog = load_tui_catalog(
            model_manifest_dir=args.model_manifest_dir,
            benchmark_manifest_dir=args.benchmark_manifest_dir,
            runtime_profile_dir=args.runtime_profile_dir,
            conda_envs_root=args.conda_envs_root,
        )
        model_id, benchmark_id = _resolve_ids(catalog, args.model_id, args.benchmark_id)
        result = _write_suite_plan(
            model_id=model_id,
            benchmark_id=benchmark_id,
            output_dir=args.output_dir / "suite-plan",
            args=args,
        )
        _safe_print(f"suite-plan: status={result.status} manifest={result.suite_manifest_path}")
        return int(result.exit_code)

    # ── Print-command mode ──────────────────────────────────────────
    if args.print_command:
        catalog = load_tui_catalog(
            model_manifest_dir=args.model_manifest_dir,
            benchmark_manifest_dir=args.benchmark_manifest_dir,
            runtime_profile_dir=args.runtime_profile_dir,
            conda_envs_root=args.conda_envs_root,
        )
        model_id, benchmark_id = _resolve_ids(catalog, args.model_id, args.benchmark_id)
        output_dir = args.output_dir / f"{model_id}__{benchmark_id}" if args.output_dir.name == "tui" else args.output_dir
        commands = _selected_commands(
            model_id=model_id,
            benchmark_id=benchmark_id,
            output_dir=output_dir,
            args=args,
        )
        _safe_print("\n".join(f"{name}: {format_shell_command(command)}" for name, command in commands))
        return 0

    # ── Fallback dashboard mode ──────────────────────────────────────
    if args.fallback:
        catalog = load_tui_catalog(
            model_manifest_dir=args.model_manifest_dir,
            benchmark_manifest_dir=args.benchmark_manifest_dir,
            runtime_profile_dir=args.runtime_profile_dir,
            conda_envs_root=args.conda_envs_root,
        )
        _safe_print(_render_fallback_dashboard(catalog, args))
        return 0

    # ── Full Textual TUI ────────────────────────────────────────────
    try:
        from .tui_app import WorldFoundryTui
    except ImportError as exc:
        # NOTE: Textual is optional — gracefully fall back when it is not installed.
        catalog = load_tui_catalog(
            model_manifest_dir=args.model_manifest_dir,
            benchmark_manifest_dir=args.benchmark_manifest_dir,
            runtime_profile_dir=args.runtime_profile_dir,
            conda_envs_root=args.conda_envs_root,
        )
        _safe_print(
            _render_fallback_dashboard(
                catalog,
                args,
                notice=(
                    "Textual is not available; showing the standard-library fallback. "
                    f"Import error: {exc}"
                ),
            )
        )
        return 0

    app = WorldFoundryTui(
        model_manifest_dir=args.model_manifest_dir,
        benchmark_manifest_dir=args.benchmark_manifest_dir,
        runtime_profile_dir=args.runtime_profile_dir,
        initial_model_id=args.model_id,
        initial_benchmark_id=args.benchmark_id,
        metrics=args.metric,
        output_dir=args.output_dir,
        input_path=args.input,
        input_dir=args.input_dir,
        video_path=args.video,
        trajectory_file=args.trajectory_file,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        task=args.task,
        mode=args.mode,
        resize_mode=args.resize_mode,
        size=args.size,
        frames=args.frames,
        steps=args.steps,
        frames_per_generation=args.frames_per_generation,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        fps=args.fps,
        dtype=args.dtype,
        max_sequence_length=args.max_sequence_length,
        cam_type=args.cam_type,
        interactions=args.interactions,
        output_formats=args.output_formats,
        trajectory=args.trajectory,
        angle=args.angle,
        distance=args.distance,
        orbit_radius=args.orbit_radius,
        zoom_ratio=args.zoom_ratio,
        alpha_threshold=args.alpha_threshold,
        static_scene=args.static_scene,
        low_vram=args.low_vram,
        disable_lora=args.disable_lora,
        vis_rendering=args.vis_rendering,
        offload_t5=args.offload_t5,
        offload_transformer_during_vae=args.offload_transformer_during_vae,
        offload_vae=args.offload_vae,
        output_path=args.output_path,
        ckpt_type=args.ckpt_type,
        ckpt_root=args.ckpt_root,
        ckpt_path=args.ckpt_path,
        conda_envs_root=args.conda_envs_root,
        gpu=args.gpu,
    )
    app.run()
    return 0


def _resolve_ids(catalog, model_id: str | None, benchmark_id: str | None) -> tuple[str, str]:
    """Resolve missing *model_id* / *benchmark_id* to the first available entry.

    Falls back to the first infer model and first benchmark when IDs are ``None``.
    Maps variant IDs to their family IDs via ``INFER_VARIANT_TO_MODEL``.

    Raises:
        SystemExit: When no models or benchmarks are available.
    """
    if model_id is None:
        if not catalog.models:
            raise SystemExit("No model-zoo entries are available.")
        infer_models = [row.model_id for row in catalog.models if is_infer_model_row(row)]
        model_id = infer_models[0] if infer_models else catalog.models[0].model_id
    infer_model_ids = {row.model_id for row in catalog.models if is_infer_model_row(row)}
    if model_id not in infer_model_ids and model_id not in INFER_VARIANT_GROUPS:
        model_id = INFER_VARIANT_TO_MODEL.get(model_id, model_id)
    if benchmark_id is None:
        if not catalog.benchmarks:
            raise SystemExit("No benchmark-zoo entries are available.")
        benchmark_id = catalog.benchmarks[0].benchmark_id
    return model_id, benchmark_id


def _safe_print(text: str) -> None:
    """Print *text* to stdout, silently swallowing ``BrokenPipeError``."""
    try:
        print(text)
    except BrokenPipeError:
        return


def _render_fallback_dashboard(catalog, args, *, notice: str | None = None) -> str:
    """Build a plain-text dashboard string for the ``--fallback`` mode.

    Args:
        catalog: The loaded :class:`TuiCatalog`.
        args: Parsed CLI namespace.
        notice: Optional message to display (e.g. Textual import error).

    Returns:
        Multi-line plain-text dashboard ready for ``_safe_print``.
    """
    model_id, benchmark_id = _resolve_ids(catalog, args.model_id, args.benchmark_id)
    output_dir = args.output_dir / f"{model_id}__{benchmark_id}" if args.output_dir.name == "tui" else args.output_dir
    commands = _selected_commands(
        model_id=model_id,
        benchmark_id=benchmark_id,
        output_dir=output_dir,
        args=args,
    )
    conda_status = dict(catalog.conda_status or {})
    lines = [
        render_fallback_header().rstrip(),
        "WorldFoundry TUI fallback",
        "=" * 40,
    ]
    if notice:
        lines.extend(["", notice])
    lines.extend(
        [
            "",
            "Catalog",
            _kv("models", len(catalog.models)),
            _kv("benchmarks", len(catalog.benchmarks)),
            "",
            "Conda",
            _kv("available", conda_status.get("conda_available")),
            _kv("executable", conda_status.get("conda_executable") or "-"),
            _kv("active env", conda_status.get("active_env") or "-"),
            _kv("active prefix", conda_status.get("active_prefix") or "-"),
            _kv("env root", conda_status.get("env_root") or "-"),
            _kv("env specs", conda_status.get("env_specs", 0)),
            _kv("envs existing", conda_status.get("envs_existing", 0)),
            "",
            "Runtime Profiles",
            *_format_rows(
                (
                    row.profile_id,
                    row.backend_stage,
                    row.integration_status,
                    row.runtime_status,
                )
                for row in catalog.runtime_profiles[:12]
            ),
            "",
            "Infer Models",
            *_format_rows(
                (
                    row.model_id,
                    _clip(",".join(label for label, _variant in infer_variant_options(row.model_id)), 44),
                )
                for row in catalog.models
                if is_infer_model_row(row)
            ),
            "",
            "Benchmarks",
            *_format_rows(
                (
                    row.benchmark_id,
                    _clip(",".join(row.domains) or "-", 24),
                    row.integration_status,
                    row.verification_status,
                )
                for row in catalog.benchmarks[:12]
            ),
            "",
            "Commands",
            *(f"{name}: {format_shell_command(command)}" for name, command in commands),
            "",
            "Suite plan command",
            f"  {format_shell_command(_suite_plan_command(model_id, benchmark_id, output_dir, args))}",
            "",
            "Benchmark run command",
            f"  {format_shell_command(_benchmark_run_command(benchmark_id, output_dir, args))}",
            "",
            "Runtime preflight command",
            f"  {format_shell_command(_runtime_preflight_command(output_dir, args))}",
            "",
            "These commands execute real infer, eval, or UI actions.",
        ]
    )
    return "\n".join(lines)


def _suite_plan_command(model_id: str, benchmark_id: str, output_dir: Path, args) -> tuple[str, ...]:
    """Build a shell command tuple that writes a suite plan for *model_id* × *benchmark_id*."""
    return build_suite_command(
        output_dir=output_dir.parent / "suite-plan",
        model_ids=(model_id,),
        benchmark_ids=(benchmark_id,),
        model_manifest_dir=args.model_manifest_dir,
        benchmark_manifest_dir=args.benchmark_manifest_dir,
        plan_only=True,
    )


def _benchmark_run_command(benchmark_id: str, output_dir: Path, args) -> tuple[str, ...]:
    """Build a shell command tuple for a selected benchmark official run."""
    command: list[str] = [
        sys.executable,
        "-m",
        "worldfoundry.cli",
        "zoo",
        "benchmark-run",
        "--benchmark-id",
        benchmark_id,
        "--mode",
        "official-run",
        "--output-dir",
        str(output_dir.parent / "benchmark-run" / benchmark_id),
        "--json",
    ]
    if args.benchmark_manifest_dir is not None:
        command.extend(["--manifest-dir", str(args.benchmark_manifest_dir)])
    return tuple(command)


def _runtime_preflight_command(output_dir: Path, args) -> tuple[str, ...]:
    """Build a shell command tuple for runtime-profile readiness checks."""
    command: list[str] = [
        sys.executable,
        "-m",
        "worldfoundry.cli",
        "preflight",
        "runtime",
        "--profile",
        "all",
        "--output-dir",
        str(output_dir.parent / "runtime-preflight"),
        "--json",
    ]
    if args.runtime_profile_dir is not None:
        command.extend(["--manifest", str(args.runtime_profile_dir)])
    return tuple(command)


def _write_suite_plan(*, model_id: str, benchmark_id: str, output_dir: Path, args):
    """Execute a model × benchmark suite plan without running cells."""
    from worldfoundry.evaluation.runner import ModelBenchmarkSuiteRequest, run_model_benchmark_suite
    from worldfoundry.evaluation.utils import BENCHMARK_ZOO_DIR, MODEL_ZOO_DIR

    return run_model_benchmark_suite(
        ModelBenchmarkSuiteRequest(
            output_dir=output_dir,
            model_manifest_dir=args.model_manifest_dir or MODEL_ZOO_DIR,
            benchmark_manifest_dir=args.benchmark_manifest_dir or BENCHMARK_ZOO_DIR,
            model_ids=(model_id,),
            benchmark_ids=(benchmark_id,),
            mode="official-run",
            execute=False,
            skip_incompatible=False,
        )
    )


def _selected_commands(
    *,
    model_id: str,
    benchmark_id: str,
    output_dir: Path,
    args,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Collect shell command tuples for the selected infer / eval / studio actions.

    Returns:
        A tuple of ``(name, command_tuple)`` pairs.
    """
    commands: list[tuple[str, tuple[str, ...]]] = []
    # NOTE: Infer command uses variant-specific control fields to filter applicable overrides
    try:
        infer_controls = set(infer_control_fields(model_id, args.ckpt_type))

        def infer_arg(field: str, value):
            """Return *value* only when *field* is applicable to the current variant."""
            return value if field in infer_controls else None

        commands.append(
            (
                "infer",
                build_model_infer_command(
                    model_id=model_id,
                    output_dir=output_dir.parent / "infer",
                    ckpt_type=args.ckpt_type,
                    ckpt_root=args.ckpt_root,
                    ckpt_path=args.ckpt_path,
                    input_path=infer_arg("input", args.input),
                    input_dir=infer_arg("input_dir", args.input_dir),
                    video_path=infer_arg("video", args.video),
                    trajectory_file=infer_arg("trajectory_file", args.trajectory_file),
                    prompt=infer_arg("prompt", args.prompt),
                    negative_prompt=infer_arg("negative_prompt", args.negative_prompt),
                    task=infer_arg("task", args.task),
                    mode=infer_arg("mode", args.mode),
                    resize_mode=infer_arg("resize_mode", args.resize_mode),
                    size=infer_arg("size", args.size),
                    frames=infer_arg("frames", args.frames),
                    steps=infer_arg("steps", args.steps),
                    frames_per_generation=infer_arg("frames_per_generation", args.frames_per_generation),
                    guidance_scale=infer_arg("guidance_scale", args.guidance_scale),
                    seed=infer_arg("seed", args.seed),
                    fps=infer_arg("fps", args.fps),
                    dtype=infer_arg("dtype", args.dtype),
                    max_sequence_length=infer_arg("max_sequence_length", args.max_sequence_length),
                    cam_type=infer_arg("cam_type", args.cam_type),
                    interactions=infer_arg("interactions", args.interactions),
                    output_formats=infer_arg("output_formats", args.output_formats),
                    trajectory=infer_arg("trajectory", args.trajectory),
                    angle=infer_arg("angle", args.angle),
                    distance=infer_arg("distance", args.distance),
                    orbit_radius=infer_arg("orbit_radius", args.orbit_radius),
                    zoom_ratio=infer_arg("zoom_ratio", args.zoom_ratio),
                    alpha_threshold=infer_arg("alpha_threshold", args.alpha_threshold),
                    static_scene=infer_arg("static_scene", args.static_scene),
                    low_vram=infer_arg("low_vram", args.low_vram),
                    disable_lora=infer_arg("disable_lora", args.disable_lora),
                    vis_rendering=infer_arg("vis_rendering", args.vis_rendering),
                    offload_t5=infer_arg("offload_t5", args.offload_t5),
                    offload_transformer_during_vae=infer_arg(
                        "offload_transformer_during_vae",
                        args.offload_transformer_during_vae,
                    ),
                    offload_vae=infer_arg("offload_vae", args.offload_vae),
                    output_path=infer_arg("output_path", args.output_path),
                    conda_envs_root=args.conda_envs_root,
                    gpu=args.gpu,
                ),
            )
        )
    except ValueError as exc:
        commands.append(("infer", ("echo", str(exc))))
    # NOTE: Eval command is always available regardless of model runner kind
    commands.append(
        (
            "eval",
            build_model_benchmark_command(
                model_id=model_id,
                benchmark_id=benchmark_id,
                output_dir=output_dir,
                model_manifest_dir=args.model_manifest_dir,
                benchmark_manifest_dir=args.benchmark_manifest_dir,
                metrics=_split_csv_values(args.metric),
            ),
        )
    )
    commands.append(("ui", build_studio_command()))
    return tuple(commands)


def _kv(key: str, value: object) -> str:
    """Format a key–value pair for the plain-text dashboard."""
    return f"  {key}: {value}"


def _format_rows(rows) -> list[str]:
    """Render an iterable of row tuples as right-aligned, column-padded text lines."""
    materialized = [tuple(str(cell) for cell in row) for row in rows]
    if not materialized:
        return ["  -"]
    widths = [max(len(row[index]) for row in materialized) for index in range(len(materialized[0]))]
    return [
        "  " + "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)).rstrip()
        for row in materialized
    ]


def _clip(value: str, limit: int) -> str:
    """Truncate *value* to *limit* characters, appending ``…`` when truncated."""
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _split_csv_values(values: Sequence[str] | None) -> tuple[str, ...]:
    """Split repeatable / comma-separated CLI values into a flat tuple of individual items."""
    items: list[str] = []
    for value in values or ():
        for item in str(value).split(","):
            item = item.strip()
            if item:
                items.append(item)
    return tuple(items)


# ── CLI bootstrap ──────────────────────────────────────────────────
if __name__ == "__main__":
    raise SystemExit(main())
