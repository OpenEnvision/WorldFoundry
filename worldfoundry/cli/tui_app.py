"""Textual TUI application for WorldFoundry inference, evaluation, and Studio.

This module defines :class:`WorldFoundryTui`, the main interactive terminal
application built on Textual. It provides model/benchmark selection,
parameter configuration, command preview, and execution logging.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
from pathlib import Path
from typing import Iterable

from rich.syntax import Syntax
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import (
    Button,
    Collapsible,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Rule,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from worldfoundry.core.io.paths import checkpoint_root_path, conda_envs_root_path

from .tui_brand import render_brand_logo
from .tui_discovery import (
    ASTRA_CAM_TYPE_OPTIONS,
    INFER_VARIANT_TO_MODEL,
    TUI_ACTIONS,
    BenchmarkCatalogRow,
    InferControlSpec,
    ModelCatalogRow,
    build_model_benchmark_command,
    build_model_infer_command,
    build_studio_command,
    format_shell_command,
    infer_control_fields,
    infer_control_specs,
    infer_model_variant_ids,
    infer_variant_label,
    infer_variant_options,
    is_infer_model_row,
    load_tui_catalog,
    resolve_infer_model_variant,
)
from .tui_icons import icons

# ── Module constants ──────────────────────────────────────────────

EVALUATION_INTENT_OPTIONS = (
    ("Model Benchmark", "model-benchmark"),
    ("Score Artifact Directory", "score-artifacts"),
)

DEFAULT_CKPT_ROOT = str(checkpoint_root_path())  # Default checkpoint root directory path.
DEFAULT_CONDA_ENVS_ROOT = str(conda_envs_root_path())  # Default Conda environments root directory path.

class WorldFoundryTui(App[None]):
    """Interactive Textual application for WorldFoundry model inference, benchmark evaluation, and Studio.

    Presents a two-column layout: the left column holds the action selector,
    model/benchmark catalog tables, and a filter input; the right column holds
    tabbed configuration panels, run controls, and a RichLog execution log.

    Attributes:
        action: Current mode — ``"infer"``, ``"eval"``, or ``"ui"``.
        selected_model_id: Active model family identifier.
        selected_benchmark_id: Active benchmark identifier.
        ckpt_type: Active checkpoint variant / function selector value.
    """

    CSS_PATH = str(Path(__file__).with_name("tui_app.tcss"))
    TITLE = f"{icons['app']} WorldFoundry"
    SUB_TITLE = "Inference · Evaluation · Studio"
    ENABLE_COMMAND_PALETTE = True
    HORIZONTAL_BREAKPOINTS = [(0, "-narrow"), (120, "-wide")]

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("c", "copy_command", "Copy cmd"),
        ("p", "preflight", "Preflight"),
        ("g", "gpu_status", "GPU"),
        ("a", "show_artifacts", "Artifacts"),
        ("ctrl+r", "run_command", "Run"),
        ("ctrl+s", "stop_command", "Stop"),
        ("f", "focus_filter", "Filter"),
        ("m", "focus_models", "Models"),
        ("t", "focus_tabs", "Settings"),
    ]

    def __init__(
        self,
        *,
        model_manifest_dir: str | Path | None = None,
        benchmark_manifest_dir: str | Path | None = None,
        runtime_profile_dir: str | Path | None = None,
        initial_model_id: str | None = None,
        initial_benchmark_id: str | None = None,
        metrics: Iterable[str] | None = None,
        output_dir: str | Path = Path("runs/tui"),
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
        static_scene: str | bool | None = None,
        low_vram: str | bool | None = None,
        disable_lora: str | bool | None = None,
        vis_rendering: str | bool | None = None,
        offload_t5: str | bool | None = None,
        offload_transformer_during_vae: str | bool | None = None,
        offload_vae: str | bool | None = None,
        output_path: str | Path | None = None,
        ckpt_type: str | None = None,
        ckpt_root: str | Path | None = None,
        ckpt_path: str | Path | None = None,
        conda_envs_root: str | Path | None = None,
        gpu: str | None = None,
    ) -> None:
        """Initialize the TUI with catalog data, default selections, and inference overrides.

        Loads the model/benchmark catalog, resolves initial selection IDs,
        and stores all inference-parameter overrides as string attributes for later
        widget binding.
        """
        super().__init__()
        # ── Catalog directories ──
        self.model_manifest_dir = Path(model_manifest_dir) if model_manifest_dir is not None else None
        self.benchmark_manifest_dir = Path(benchmark_manifest_dir) if benchmark_manifest_dir is not None else None
        self.runtime_profile_dir = Path(runtime_profile_dir) if runtime_profile_dir is not None else None
        self.conda_envs_root = str(
            conda_envs_root or os.environ.get("WORLDFOUNDRY_CONDA_ENVS_ROOT") or DEFAULT_CONDA_ENVS_ROOT
        )
        # ── Load catalog ──
        self.catalog = load_tui_catalog(
            model_manifest_dir=self.model_manifest_dir,
            benchmark_manifest_dir=self.benchmark_manifest_dir,
            runtime_profile_dir=self.runtime_profile_dir,
            conda_envs_root=self.conda_envs_root,
        )
        self.model_rows = list(self.catalog.models)
        self.benchmark_rows = list(self.catalog.benchmarks)
        # ── Resolve initial selections ──
        infer_model_ids = [row.model_id for row in self.model_rows if is_infer_model_row(row)]
        initial_family_id = INFER_VARIANT_TO_MODEL.get(initial_model_id or "", initial_model_id or "")
        self.selected_model_id = (
            initial_family_id
            if initial_family_id in set(infer_model_ids)
            else (infer_model_ids[0] if infer_model_ids else (self.model_rows[0].model_id if self.model_rows else ""))
        )
        self.selected_benchmark_id = initial_benchmark_id or (
            self.benchmark_rows[0].benchmark_id if self.benchmark_rows else ""
        )
        # ── Action mode and evaluation config ──
        self.action = "infer"
        self.eval_metrics = ",".join(_split_csv_values(metrics))
        self.eval_intent = "model-benchmark"
        self.eval_artifact_dir = ""
        # ── Inference parameter overrides ──
        self.input_path = "" if input_path is None else str(input_path)
        self.input_dir = "" if input_dir is None else str(input_dir)
        self.video_path = "" if video_path is None else str(video_path)
        self.trajectory_file = "" if trajectory_file is None else str(trajectory_file)
        self.prompt = "" if prompt is None else str(prompt)
        self.negative_prompt = "" if negative_prompt is None else str(negative_prompt)
        self.infer_task = "" if task is None else str(task)
        self.infer_mode = "" if mode is None else str(mode)
        self.infer_resize_mode = "" if resize_mode is None else str(resize_mode)
        self.infer_size = "" if size is None else str(size)
        self.infer_frames = "" if frames is None else str(frames)
        self.infer_steps = "" if steps is None else str(steps)
        self.infer_frames_per_generation = "" if frames_per_generation is None else str(frames_per_generation)
        self.infer_guidance_scale = "" if guidance_scale is None else str(guidance_scale)
        self.infer_seed = "" if seed is None else str(seed)
        self.infer_fps = "" if fps is None else str(fps)
        self.infer_dtype = "" if dtype is None else str(dtype)
        self.infer_max_sequence_length = "" if max_sequence_length is None else str(max_sequence_length)
        self.infer_cam_type = str(cam_type or "4")
        self.infer_interactions = "" if interactions is None else str(interactions)
        self.infer_output_formats = "" if output_formats is None else str(output_formats)
        self.infer_trajectory = "" if trajectory is None else str(trajectory)
        self.infer_angle = "" if angle is None else str(angle)
        self.infer_distance = "" if distance is None else str(distance)
        self.infer_orbit_radius = "" if orbit_radius is None else str(orbit_radius)
        self.infer_zoom_ratio = "" if zoom_ratio is None else str(zoom_ratio)
        self.infer_alpha_threshold = "" if alpha_threshold is None else str(alpha_threshold)
        self.infer_static_scene = "" if static_scene is None else str(static_scene)
        self.infer_low_vram = "" if low_vram is None else str(low_vram)
        self.infer_disable_lora = "" if disable_lora is None else str(disable_lora)
        self.infer_vis_rendering = "" if vis_rendering is None else str(vis_rendering)
        self.infer_offload_t5 = "" if offload_t5 is None else str(offload_t5)
        self.infer_offload_transformer_during_vae = (
            "" if offload_transformer_during_vae is None else str(offload_transformer_during_vae)
        )
        self.infer_offload_vae = "" if offload_vae is None else str(offload_vae)
        # ── Paths and runtime ──
        self.output_path = "" if output_path is None else str(output_path)
        self.ckpt_type = str(ckpt_type or (initial_model_id if initial_model_id in INFER_VARIANT_TO_MODEL else ""))
        self._sync_ckpt_type_for_selected_model()
        self.ckpt_root = str(ckpt_root or os.environ.get("WORLDFOUNDRY_CKPT_DIR") or DEFAULT_CKPT_ROOT)
        self.ckpt_path = "" if ckpt_path is None else str(ckpt_path)
        self.gpu = gpu if gpu is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "")
        self.base_output_dir = Path(output_dir)
        # ── Runtime state ──
        self.output_dir_touched = False
        self.command_preview_visible = False
        self.running_process: asyncio.subprocess.Process | None = None
        self.running_task: asyncio.Task[None] | None = None
        self.stop_requested = False

    def compose(self) -> ComposeResult:
        """Build the widget tree for the TUI layout.

        Yields a two-column layout with catalog tables on the left and
        configuration tabs, run controls, and a log panel on the right.
        """
        yield Header(show_clock=True)
        with Horizontal(id="main-content"):
            with Vertical(id="left-column"):
                with Vertical(id="status-pane", classes="pane"):
                    with Collapsible(title="OpenEnvision", id="brand-collapse", collapsed=False):
                        yield Static(render_brand_logo(width_chars=64), id="brand-logo", markup=True)
                    with Horizontal(id="action-row"):
                        yield Label("Mode:", id="action-label")
                        yield Select(icons.action_options(), value=self.action, id="action", allow_blank=False)
                    yield Rule(line_style="dashed", id="status-divider")
                    yield Static("", id="conda-status")
                with Vertical(id="catalogs-pane", classes="pane"):
                    with Horizontal(id="search-row"):
                        yield Input(placeholder="Filter models and benchmarks", id="filter")
                    with Vertical(id="models-pane"):
                        yield Label(icons["models"], classes="pane-title", id="models-pane-title")
                        yield DataTable(id="models-table", cursor_type="row", zebra_stripes=True)
                    with Vertical(id="benchmarks-pane"):
                        yield Label(icons["benchmarks"], classes="pane-title", id="benchmarks-pane-title")
                        yield DataTable(id="benchmarks-table", cursor_type="row", zebra_stripes=True)

            with VerticalScroll(id="right-column"):
                yield Static("", id="selection-summary", markup=True)
                with Vertical(id="run-builder"):
                    with Horizontal(id="run-controls"):
                        yield Button(icons["run"], id="run", variant="success")
                        yield Button(icons["stop"], id="stop", variant="error")
                        with Horizontal(id="secondary-actions"):
                            yield Button(icons["copy"], id="copy", variant="primary")
                            yield Button(icons["cmd"], id="toggle-command")
                            yield Button(icons["check"], id="preflight")
                            yield Button(icons["files"], id="artifacts")
                            yield Button(icons["gpu"], id="gpu-status")
                            yield Button(icons["sync"], id="refresh")
                    with Collapsible(title=icons["command_preview"], collapsed=True, id="command-preview"):
                        yield Static("", id="command")
                    with TabbedContent(id="config-tabs", initial="tab-general"):
                        with TabPane(icons["general"], id="tab-general"):
                            with VerticalScroll(classes="tab-scroll"):
                                yield Static(icons["prompt_media"], classes="section-title", id="section-prompt-title")
                                yield Rule(line_style="heavy", classes="section-rule", id="section-prompt-rule")
                                with Vertical(id="prompt-controls", classes="field-group"):
                                    yield Label("Prompt", id="prompt-label", classes="field-caption")
                                    yield Input(value=self.prompt, placeholder="Describe the scene or action…", id="prompt")
                                with Vertical(id="negative-prompt-controls", classes="field-group"):
                                    yield Label("Negative prompt", id="negative-prompt-label", classes="field-caption")
                                    yield Input(value=self.negative_prompt, placeholder="Optional negative prompt", id="negative-prompt")
                                with Grid(id="media-controls", classes="field-grid"):
                                    with Vertical(classes="field-group"):
                                        yield Label("Input path", id="input-path-label", classes="field-caption")
                                        yield Input(value=self.input_path, placeholder="image/video path (drag & drop)", id="input-path")
                                    with Vertical(classes="field-group"):
                                        yield Label("Video path", id="video-path-label", classes="field-caption")
                                        yield Input(value=self.video_path, placeholder="optional video path (drag & drop)", id="video-path")
                                with Grid(id="flashworld-controls", classes="field-grid"):
                                    with Vertical(classes="field-group"):
                                        yield Label("JSON directory", id="input-dir-label", classes="field-caption")
                                        yield Input(value=self.input_dir, placeholder="FlashWorld JSON dir", id="input-dir")
                                    with Vertical(classes="field-group"):
                                        yield Label("Output formats", id="output-formats-label", classes="field-caption")
                                        yield Input(value=self.infer_output_formats, placeholder="video,spz,ply", id="output-formats")
                                with Grid(id="task-controls", classes="field-grid"):
                                    with Vertical(classes="field-group"):
                                        yield Label("Task", id="task-label", classes="field-caption")
                                        yield Input(value=self.infer_task, placeholder="task override", id="task")
                                    with Vertical(classes="field-group"):
                                        yield Label("Mode", id="mode-label", classes="field-caption")
                                        yield Input(value=self.infer_mode, placeholder="mode override", id="mode")
                                with Vertical(id="eval-controls", classes="field-group"):
                                    yield Label("Evaluation intent", classes="field-caption")
                                    yield Select(
                                        EVALUATION_INTENT_OPTIONS,
                                        value=self.eval_intent,
                                        allow_blank=False,
                                        id="eval-intent",
                                    )
                                    with Vertical(id="eval-artifact-controls", classes="field-group"):
                                        yield Label("Artifact directory", classes="field-caption")
                                        yield Input(
                                            value=self.eval_artifact_dir,
                                            placeholder="directory containing generated artifacts",
                                            id="eval-artifact-dir",
                                        )
                                    with Vertical(id="eval-metrics-controls", classes="field-group"):
                                        yield Label("Metrics", id="metrics-label", classes="field-caption")
                                        yield Input(
                                            value=self.eval_metrics,
                                            placeholder="artifact_count,required_artifacts_present",
                                            id="metrics",
                                        )
                                yield Static(icons["paths_runtime"], classes="section-title", id="section-paths-title")
                                yield Rule(line_style="heavy", classes="section-rule", id="section-paths-rule")
                                with Grid(id="path-controls", classes="field-grid"):
                                    with Vertical(classes="field-group"):
                                        yield Label("Output directory", classes="field-caption")
                                        yield Input(
                                            value=str(self._default_output_dir()),
                                            placeholder="runs/tui/…",
                                            id="output-dir",
                                        )
                                    with Vertical(classes="field-group"):
                                        yield Label("Artifact path", id="output-path-label", classes="field-caption")
                                        yield Input(
                                            value=self.output_path,
                                            placeholder="optional exact artifact path",
                                            id="output-path",
                                        )
                                    with Vertical(id="runtime-controls", classes="field-group full-width"):
                                        yield Label("Checkpoint root", id="ckpt-root-label", classes="field-caption")
                                        yield Input(value=self.ckpt_root, placeholder="checkpoint root", id="ckpt-root")
                                    with Vertical(classes="field-group"):
                                        yield Label("Conda env root", id="conda-envs-root-label", classes="field-caption")
                                        yield Input(value=self.conda_envs_root, placeholder="conda envs root", id="conda-envs-root")
                                    with Vertical(classes="field-group"):
                                        yield Label("GPU ids", id="gpu-label", classes="field-caption")
                                        yield Input(value=self.gpu, placeholder="0 or 0,1", id="gpu")
                                yield Static(
                                    "No general options for this selection.",
                                    id="tab-general-empty",
                                    classes="tab-empty",
                                )
                        with TabPane(icons["sampling"], id="tab-sampling"):
                            with VerticalScroll(classes="tab-scroll"):
                                with Grid(id="sampling-controls", classes="field-grid"):
                                    with Vertical(classes="field-group"):
                                        yield Label("Size", id="size-label", classes="field-caption")
                                        yield Input(value=self.infer_size, placeholder="832*480", id="size")
                                    with Vertical(classes="field-group"):
                                        yield Label("Frames", id="frames-label", classes="field-caption")
                                        yield Input(value=self.infer_frames, placeholder="frame count", id="frames")
                                    with Vertical(classes="field-group"):
                                        yield Label("Steps", id="steps-label", classes="field-caption")
                                        yield Input(value=self.infer_steps, placeholder="sampling steps", id="steps")
                                    with Vertical(classes="field-group"):
                                        yield Label("Seed", id="seed-label", classes="field-caption")
                                        yield Input(value=self.infer_seed, placeholder="random seed", id="seed")
                                with Grid(id="sampling-controls-b", classes="field-grid"):
                                    with Vertical(classes="field-group"):
                                        yield Label("FPS", id="fps-label", classes="field-caption")
                                        yield Input(value=self.infer_fps, placeholder="output fps", id="fps")
                                    with Vertical(classes="field-group"):
                                        yield Label("Guidance scale", id="guidance-scale-label", classes="field-caption")
                                        yield Input(value=self.infer_guidance_scale, placeholder="CFG / guidance", id="guidance-scale")
                                    with Vertical(classes="field-group"):
                                        yield Label("Dtype", id="dtype-label", classes="field-caption")
                                        yield Input(value=self.infer_dtype, placeholder="float16 / bfloat16", id="dtype")
                                    with Vertical(classes="field-group"):
                                        yield Label("Max sequence", id="max-sequence-length-label", classes="field-caption")
                                        yield Input(value=self.infer_max_sequence_length, placeholder="text encoder length", id="max-sequence-length")
                                yield Static("No sampling options for this model.", id="tab-sampling-empty", classes="tab-empty")
                        with TabPane(icons["camera"], id="tab-camera"):
                            with VerticalScroll(classes="tab-scroll"):
                                with Grid(id="interaction-controls", classes="field-grid"):
                                    with Vertical(classes="field-group"):
                                        yield Label("Actions", id="interactions-label", classes="field-caption")
                                        yield Input(value=self.infer_interactions, placeholder="forward,camera_l,…", id="interactions")
                                    with Vertical(classes="field-group"):
                                        yield Label("Trajectory", id="trajectory-label", classes="field-caption")
                                        yield Input(value=self.infer_trajectory, placeholder="tilt_up / move_right", id="trajectory")
                                with Grid(id="trajectory-file-controls", classes="field-grid"):
                                    with Vertical(classes="field-group"):
                                        yield Label("Trajectory file", id="trajectory-file-label", classes="field-caption")
                                        yield Input(value=self.trajectory_file, placeholder="custom trajectory JSON", id="trajectory-file")
                                    with Vertical(classes="field-group"):
                                        yield Label("Resize mode", id="resize-mode-label", classes="field-caption")
                                        yield Input(value=self.infer_resize_mode, placeholder="center_crop", id="resize-mode")
                                with Grid(id="astra-controls", classes="field-grid"):
                                    with Vertical(classes="field-group"):
                                        yield Label("Camera type", id="cam-type-label", classes="field-caption")
                                        yield Select(
                                            ASTRA_CAM_TYPE_OPTIONS,
                                            value=self.infer_cam_type
                                            if self.infer_cam_type in {value for _label, value in ASTRA_CAM_TYPE_OPTIONS}
                                            else "4",
                                            allow_blank=False,
                                            id="cam-type",
                                        )
                                    with Vertical(classes="field-group"):
                                        yield Label("Frame chunk", id="frames-per-generation-label", classes="field-caption")
                                        yield Input(value=self.infer_frames_per_generation, placeholder="8", id="frames-per-generation")
                                with Grid(id="camera-controls", classes="field-grid"):
                                    with Vertical(classes="field-group"):
                                        yield Label("Angle", id="angle-label", classes="field-caption")
                                        yield Input(value=self.infer_angle, placeholder="15", id="angle")
                                    with Vertical(classes="field-group"):
                                        yield Label("Distance", id="distance-label", classes="field-caption")
                                        yield Input(value=self.infer_distance, placeholder="0.1", id="distance")
                                    with Vertical(classes="field-group"):
                                        yield Label("Orbit radius", id="orbit-radius-label", classes="field-caption")
                                        yield Input(value=self.infer_orbit_radius, placeholder="1.0", id="orbit-radius")
                                    with Vertical(classes="field-group"):
                                        yield Label("Zoom ratio", id="zoom-ratio-label", classes="field-caption")
                                        yield Input(value=self.infer_zoom_ratio, placeholder="1.0", id="zoom-ratio")
                                yield Static("No camera options for this model.", id="tab-camera-empty", classes="tab-empty")
                        with TabPane(icons["model"], id="tab-model"):
                            with VerticalScroll(classes="tab-scroll"):
                                with Vertical(id="checkpoint-controls", classes="field-group"):
                                    yield Label("Function / variant", classes="field-caption")
                                    yield Select(
                                        self._ckpt_type_options(),
                                        value=self.ckpt_type,
                                        allow_blank=False,
                                        id="ckpt-type",
                                    )
                                with Vertical(id="checkpoint-controls-b", classes="field-group"):
                                    yield Label("Checkpoint override", classes="field-caption")
                                    yield Input(value=self.ckpt_path, placeholder="optional exact checkpoint path", id="ckpt-path")
                                yield Static(icons["advanced"], classes="section-title", id="section-model-title")
                                yield Rule(line_style="heavy", classes="section-rule", id="section-model-rule")
                                with Grid(id="neoverse-controls", classes="field-grid"):
                                    with Vertical(classes="field-group"):
                                        yield Label("Alpha threshold", id="alpha-threshold-label", classes="field-caption")
                                        yield Input(value=self.infer_alpha_threshold, placeholder="1.0", id="alpha-threshold")
                                    with Vertical(classes="field-group"):
                                        yield Label("Static scene", id="static-scene-label", classes="field-caption")
                                        yield Input(value=self.infer_static_scene, placeholder="1 / 0", id="static-scene")
                                with Grid(id="neoverse-controls-b", classes="field-grid"):
                                    with Vertical(classes="field-group"):
                                        yield Label("Low VRAM", id="low-vram-label", classes="field-caption")
                                        yield Input(value=self.infer_low_vram, placeholder="1 / 0", id="low-vram")
                                    with Vertical(classes="field-group"):
                                        yield Label("Disable LoRA", id="disable-lora-label", classes="field-caption")
                                        yield Input(value=self.infer_disable_lora, placeholder="1 / 0", id="disable-lora")
                                    with Vertical(classes="field-group"):
                                        yield Label("Vis rendering", id="vis-rendering-label", classes="field-caption")
                                        yield Input(value=self.infer_vis_rendering, placeholder="1 / 0", id="vis-rendering")
                                with Grid(id="flashworld-offload-controls", classes="field-grid"):
                                    with Vertical(classes="field-group"):
                                        yield Label("Offload T5", id="offload-t5-label", classes="field-caption")
                                        yield Input(value=self.infer_offload_t5, placeholder="1 / 0", id="offload-t5")
                                    with Vertical(classes="field-group"):
                                        yield Label("Offload transformer", id="offload-transformer-during-vae-label", classes="field-caption")
                                        yield Input(
                                            value=self.infer_offload_transformer_during_vae,
                                            placeholder="1 / 0",
                                            id="offload-transformer-during-vae",
                                        )
                                    with Vertical(classes="field-group"):
                                        yield Label("Offload VAE", id="offload-vae-label", classes="field-caption")
                                        yield Input(value=self.infer_offload_vae, placeholder="1 / 0", id="offload-vae")
                                yield Static("No model-specific options.", id="tab-model-empty", classes="tab-empty")
                    with Vertical(id="studio-panel"):
                        yield Label(icons["studio_title"], id="studio-help-title")
                        yield Static(
                            f"Launch the local Studio web UI for interactive model and benchmark exploration.\n\n"
                            f"Press [bold]{icons['run']}[/] to start the server. Logs appear below.",
                            id="studio-help",
                            markup=True,
                        )
                    with Vertical(id="status-bar"):
                        yield Static("", id="selection-status")
                with Vertical(id="log-panel"):
                    yield RichLog(id="logs", highlight=True, markup=True, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        """Initialize tables, sync layouts, and focus the filter input on startup."""
        # ── Apply initial theme ──
        self._sync_action_theme()
        self._sync_brand_logo()
        
        # ── Set bpytop-style border titles ──
        self.query_one("#status-pane", Vertical).border_title = " OpenEnvision "
        self.query_one("#catalogs-pane", Vertical).border_title = " Catalogs "
        self.query_one("#run-builder", Vertical).border_title = " Configuration "
        self.query_one("#log-panel", Vertical).border_title = " Execution Log "
        
        self.query_one("#selection-summary", Static).border_title = " Overview "
        if self.size.width < 120 or self.size.height < 35:
            self.query_one("#brand-collapse", Collapsible).collapsed = True
        # ── Configure DataTables ──
        for table_id in ("models-table", "benchmarks-table"):
            table = self.query_one(f"#{table_id}", DataTable)
            table.fixed_columns = 1
            table.cursor_type = "row"
        
        # ── Sync all UI state ──
        self._sync_action_select()
        self._sync_action_layout()
        self._sync_command_preview_layout()
        self._sync_infer_control_layout()
        self._sync_run_controls()
        self._populate_tables()
        self._update_command()
        self._write_log(
            f"[bold green]{icons['ready']}[/] · catalog loaded · "
            "[dim]Ctrl+Q[/] quit · [dim]r[/] refresh · [dim]c[/] copy · "
            "[dim]Tab[/] / [dim]Shift+Tab[/] navigate"
        )
        # NOTE: Show font-setup hint if the icon mode needs user intervention
        hint = icons.startup_hint()
        if hint:
            self._write_log(hint)

        # NOTE: Automatically focus the filter input on startup for smooth keyboard usage
        self.query_one("#filter", Input).focus()

    def on_resize(self, event: events.Resize) -> None:
        """Re-render the brand logo when the terminal size changes."""
        if event.size.width < 120 or event.size.height < 35:
            try:
                self.query_one("#brand-collapse", Collapsible).collapsed = True
            except NoMatches:
                pass
        self._sync_brand_logo()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Sync internal state and command preview when any input widget changes."""
        if event.input.id == "filter":
            self._populate_tables(event.value)
            self._collapse_brand()
        elif event.input.id == "output-dir":
            # Initial composition and mode/model changes also assign this
            # value.  Only preserve it as a user override while the field is
            # actively focused by the user.
            if event.input.has_focus:
                self.output_dir_touched = True
            self._update_command()
        elif event.input.id == "input-path":
            self.input_path = event.value.strip()
            self._update_command()
        elif event.input.id == "input-dir":
            self.input_dir = event.value.strip()
            self._update_command()
        elif event.input.id == "video-path":
            self.video_path = event.value.strip()
            self._update_command()
        elif event.input.id == "trajectory-file":
            self.trajectory_file = event.value.strip()
            self._update_command()
        elif event.input.id == "prompt":
            self.prompt = event.value.strip()
            self._update_command()
        elif event.input.id == "negative-prompt":
            self.negative_prompt = event.value.strip()
            self._update_command()
        elif event.input.id == "task":
            self.infer_task = event.value.strip()
            self._update_command()
        elif event.input.id == "mode":
            self.infer_mode = event.value.strip()
            self._update_command()
        elif event.input.id == "resize-mode":
            self.infer_resize_mode = event.value.strip()
            self._update_command()
        elif event.input.id == "size":
            self.infer_size = event.value.strip()
            self._update_command()
        elif event.input.id == "frames":
            self.infer_frames = event.value.strip()
            self._update_command()
        elif event.input.id == "steps":
            self.infer_steps = event.value.strip()
            self._update_command()
        elif event.input.id == "frames-per-generation":
            self.infer_frames_per_generation = event.value.strip()
            self._update_command()
        elif event.input.id == "guidance-scale":
            self.infer_guidance_scale = event.value.strip()
            self._update_command()
        elif event.input.id == "seed":
            self.infer_seed = event.value.strip()
            self._update_command()
        elif event.input.id == "fps":
            self.infer_fps = event.value.strip()
            self._update_command()
        elif event.input.id == "dtype":
            self.infer_dtype = event.value.strip()
            self._update_command()
        elif event.input.id == "max-sequence-length":
            self.infer_max_sequence_length = event.value.strip()
            self._update_command()
        elif event.input.id == "interactions":
            self.infer_interactions = event.value.strip()
            self._update_command()
        elif event.input.id == "output-formats":
            self.infer_output_formats = event.value.strip()
            self._update_command()
        elif event.input.id == "metrics":
            self.eval_metrics = event.value.strip()
            self._update_command()
        elif event.input.id == "eval-artifact-dir":
            self.eval_artifact_dir = event.value.strip()
            self._update_command()
        elif event.input.id == "trajectory":
            self.infer_trajectory = event.value.strip()
            self._update_command()
        elif event.input.id == "angle":
            self.infer_angle = event.value.strip()
            self._update_command()
        elif event.input.id == "distance":
            self.infer_distance = event.value.strip()
            self._update_command()
        elif event.input.id == "orbit-radius":
            self.infer_orbit_radius = event.value.strip()
            self._update_command()
        elif event.input.id == "zoom-ratio":
            self.infer_zoom_ratio = event.value.strip()
            self._update_command()
        elif event.input.id == "alpha-threshold":
            self.infer_alpha_threshold = event.value.strip()
            self._update_command()
        elif event.input.id == "static-scene":
            self.infer_static_scene = event.value.strip()
            self._update_command()
        elif event.input.id == "low-vram":
            self.infer_low_vram = event.value.strip()
            self._update_command()
        elif event.input.id == "disable-lora":
            self.infer_disable_lora = event.value.strip()
            self._update_command()
        elif event.input.id == "vis-rendering":
            self.infer_vis_rendering = event.value.strip()
            self._update_command()
        elif event.input.id == "offload-t5":
            self.infer_offload_t5 = event.value.strip()
            self._update_command()
        elif event.input.id == "offload-transformer-during-vae":
            self.infer_offload_transformer_during_vae = event.value.strip()
            self._update_command()
        elif event.input.id == "offload-vae":
            self.infer_offload_vae = event.value.strip()
            self._update_command()
        elif event.input.id == "output-path":
            self.output_path = event.value.strip()
            self._update_command()
        elif event.input.id == "ckpt-root":
            self.ckpt_root = event.value.strip()
            self._update_command()
        elif event.input.id == "ckpt-path":
            self.ckpt_path = event.value.strip()
            self._update_command()
        elif event.input.id == "conda-envs-root":
            self.conda_envs_root = event.value.strip()
            self._update_command()
        elif event.input.id == "gpu":
            self.gpu = event.value.strip()
            self._update_command()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select-widget changes for action mode, checkpoint type, and camera options."""
        if event.select.id == "action":
            action = _normalize_select_value(event.value, allowed=TUI_ACTIONS)
            if action is None:
                self._sync_action_select()
                return
            self.action = action
            self._sync_action_theme()

            try:
                collapse = self.query_one("#brand-collapse", Collapsible)
                collapse.collapsed = self.size.width < 120 or self.size.height < 35
            except NoMatches:
                pass

            selected_row = next((row for row in self.model_rows if row.model_id == self.selected_model_id), None)
            if self.action == "infer" and (selected_row is None or not is_infer_model_row(selected_row)):
                infer_rows = [row for row in self.model_rows if is_infer_model_row(row)]
                if infer_rows:
                    self.selected_model_id = infer_rows[0].model_id
            elif self.action == "eval" and selected_row is not None and selected_row.runner_kind in {"infer_script", "studio_runtime"}:
                eval_rows = [row for row in self.model_rows if row.runner_kind not in {"infer_script", "studio_runtime"}]
                if eval_rows:
                    self.selected_model_id = eval_rows[0].model_id
            self._sync_ckpt_type_for_selected_model()
            self._refresh_ckpt_type_select()
            self._sync_action_layout()
            self._sync_infer_control_layout()
            if not self.output_dir_touched:
                self.query_one("#output-dir", Input).value = str(self._default_output_dir())
            self._populate_tables(self.query_one("#filter", Input).value)
            self._update_command()
        elif event.select.id == "ckpt-type":
            ckpt_type = _normalize_select_value(event.value)
            if ckpt_type is None:
                return
            self.ckpt_type = ckpt_type
            self._sync_infer_control_layout()
            self._update_command()
        elif event.select.id == "eval-intent":
            intent = _normalize_select_value(
                event.value,
                allowed={value for _label, value in EVALUATION_INTENT_OPTIONS},
            )
            if intent is None:
                return
            self.eval_intent = intent
            self._sync_action_layout()
            if not self.output_dir_touched:
                self.query_one("#output-dir", Input).value = str(self._default_output_dir())
            self._update_command()
        elif event.select.id == "cam-type":
            cam_type = _normalize_select_value(event.value)
            if cam_type is None:
                return
            self.infer_cam_type = cam_type
            self._update_command()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dispatch button presses to the corresponding run, stop, copy, or utility action."""
        button_id = event.button.id
        if button_id == "run":
            self._start_selected_command()
        elif button_id == "stop":
            await self._stop_process()
        elif button_id == "copy":
            await self.action_copy_command()
        elif button_id == "toggle-command":
            self.action_toggle_command_preview()
        elif button_id == "preflight":
            self._run_preflight()
        elif button_id == "artifacts":
            self._show_artifacts()
        elif button_id == "gpu-status":
            await self._log_gpu_status()
        elif button_id == "refresh":
            self.action_refresh()

    async def on_unmount(self) -> None:
        """Terminate the running subprocess and cancel the task on app exit."""
        task = self.running_task
        process = self.running_process
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _collapse_brand(self) -> None:
        """Collapse the brand logo panel to save vertical space."""
        try:
            collapse = self.query_one("#brand-collapse", Collapsible)
            collapse.collapsed = True
        except NoMatches:
            pass

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Update the selected model or benchmark when a catalog row is clicked."""
        self._collapse_brand()
        row_key = getattr(event.row_key, "value", event.row_key)
        if event.data_table.id == "models-table":
            self.selected_model_id = str(row_key)
            self._sync_ckpt_type_for_selected_model()
            self._refresh_ckpt_type_select()
            self._sync_infer_control_layout()
        elif event.data_table.id == "benchmarks-table":
            self.selected_benchmark_id = str(row_key)
        if not self.output_dir_touched:
            output_input = self.query_one("#output-dir", Input)
            output_input.value = str(self._default_output_dir())
        self._update_command()

    def action_focus_filter(self) -> None:
        """Focus the catalog filter input."""
        self.query_one("#filter", Input).focus()

    def action_focus_models(self) -> None:
        """Focus the models catalog table."""
        table = self.query_one("#models-table", DataTable)
        table.focus()

    def action_focus_tabs(self) -> None:
        """Focus the configuration tabbed content panel."""
        try:
            self.query_one("#config-tabs", TabbedContent).focus()
        except NoMatches:
            pass

    def action_run_command(self) -> None:
        """Start the selected command if the Run button is not disabled."""
        if not self.query_one("#run", Button).disabled:
            self._start_selected_command()

    def action_stop_command(self) -> None:
        """Stop the running subprocess if the Stop button is not disabled."""
        if not self.query_one("#stop", Button).disabled:
            asyncio.create_task(self._stop_process())

    def action_refresh(self) -> None:
        """Reload the catalog from disk and re-sync all UI state."""
        try:
            self.catalog = load_tui_catalog(
                model_manifest_dir=self.model_manifest_dir,
                benchmark_manifest_dir=self.benchmark_manifest_dir,
                runtime_profile_dir=self.runtime_profile_dir,
                conda_envs_root=self.conda_envs_root,
            )
        except Exception as exc:  # noqa: BLE001 - keep the interactive app alive and surface the failure in its log.
            self._write_log(f"Catalog refresh failed: {type(exc).__name__}: {exc}")
            return
        self.model_rows = list(self.catalog.models)
        self.benchmark_rows = list(self.catalog.benchmarks)
        visible_model_ids = {row.model_id for row in self.model_rows if self._model_visible_for_action(row)}
        if self.selected_model_id not in visible_model_ids and self.model_rows:
            self.selected_model_id = next(iter(visible_model_ids), self.model_rows[0].model_id)
        self._sync_ckpt_type_for_selected_model()
        if self.selected_benchmark_id not in {row.benchmark_id for row in self.benchmark_rows} and self.benchmark_rows:
            self.selected_benchmark_id = self.benchmark_rows[0].benchmark_id
        self._populate_tables(self.query_one("#filter", Input).value)
        self._refresh_ckpt_type_select()
        self._sync_action_layout()
        self._sync_infer_control_layout()
        self._update_command()
        self._write_log("Catalog refreshed.")

    async def action_copy_command(self) -> None:
        """Copy the current command text to the system clipboard."""
        command = self._command_text()
        copied = False
        if hasattr(self, "copy_to_clipboard"):
            self.copy_to_clipboard(command)
            copied = True
        elif hasattr(self, "set_clipboard"):
            self.set_clipboard(command)  # type: ignore[attr-defined]
            copied = True
        self._write_log("Command copied to clipboard." if copied else "Clipboard API is not available in this terminal.")

    def action_toggle_command_preview(self) -> None:
        """Toggle visibility of the command-preview collapsible panel."""
        self.command_preview_visible = not self.command_preview_visible
        self._sync_command_preview_layout()
        self._update_command()

    def action_preflight(self) -> None:
        """Run a preflight check and log the results."""
        self._run_preflight()

    async def action_gpu_status(self) -> None:
        """Query GPU compute processes via ``nvidia-smi`` and log the result."""
        await self._log_gpu_status()

    def action_show_artifacts(self) -> None:
        """List recent output artifacts from the current output directory."""
        self._show_artifacts()

    def _run_preflight(self) -> None:
        """Log a preflight summary of paths, environment, and readiness checks."""
        self._write_log("Preflight")
        self._write_log("$ " + self._command_text())
        for line in self._preflight_lines():
            self._write_log(line)

    def _show_artifacts(self) -> None:
        """List the most recent output artifacts from the current run directory."""
        output_dir = self._current_output_dir()
        self._write_log(f"Artifacts: {output_dir}")
        if not output_dir.exists():
            self._write_log("  output directory does not exist yet")
            return
        if output_dir.is_file():
            self._write_log(f"  {_format_file_artifact(output_dir)}")
            return
        files = sorted((path for path in output_dir.rglob("*") if path.is_file()), key=lambda path: path.stat().st_mtime, reverse=True)
        if not files:
            self._write_log("  no files written yet")
            return
        for path in files[:30]:
            self._write_log(f"  {_format_file_artifact(path, root=output_dir)}")
        if len(files) > 30:
            self._write_log(f"  ... {len(files) - 30} more files")

    async def _log_gpu_status(self) -> None:
        if shutil.which("nvidia-smi") is None:
            self._write_log("nvidia-smi is not available.")
            return
        command = (
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        )
        self._write_log("$ " + format_shell_command(command))
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert process.stdout is not None
        lines: list[str] = []
        async for raw_line in process.stdout:
            line = raw_line.decode(errors="replace").strip()
            if line:
                lines.append(line)
        exit_code = await process.wait()
        if exit_code != 0:
            self._write_log(f"GPU status command exited with code {exit_code}.")
            return
        if not lines:
            self._write_log("No running GPU compute processes.")
            return
        for line in lines:
            self._write_log("GPU " + line)

    def _populate_tables(self, query: str = "") -> None:
        """Filter and repopulate the model / benchmark tables from the catalog."""
        query = query.strip().casefold()
        models = [row for row in self.model_rows if self._model_visible_for_action(row) and self._matches_model(row, query)]
        benchmarks = [row for row in self.benchmark_rows if self._matches_benchmark(row, query)]
        self._populate_model_table(models)
        if self.action == "eval":
            self._populate_benchmark_table(benchmarks)
        self._update_pane_titles(len(models), len(benchmarks))
        self._update_selection_summary(len(models), len(benchmarks))

    def _populate_model_table(self, rows: Iterable[ModelCatalogRow]) -> None:
        """Clear and repopulate the models ``DataTable`` with the given rows."""
        table = self.query_one("#models-table", DataTable)
        table.clear(columns=True)
        table.add_columns("ID", "Task", "Functions", "Status", "Run", "VRAM")
        for row in rows:
            weight_types = tuple(infer_variant_label(variant) for variant in infer_model_variant_ids(row.model_id))
            table.add_row(
                row.model_id,
                _short_join(row.tasks, limit=34),
                _short_join(weight_types, limit=34),
                _format_status(row.integration_status),
                "infer" if is_infer_model_row(row) else row.runner_kind,
                "-" if row.min_vram_gb is None else f"{row.min_vram_gb:g}G",
                key=row.model_id,
            )

    def _populate_benchmark_table(self, rows: Iterable[BenchmarkCatalogRow]) -> None:
        """Clear and repopulate the benchmarks ``DataTable`` with the given rows."""
        table = self.query_one("#benchmarks-table", DataTable)
        table.clear(columns=True)
        table.add_columns("ID", "Domain", "Modality", "Status", "Verify", "Metrics")
        for row in rows:
            table.add_row(
                row.benchmark_id,
                _short_join(row.domains, limit=30),
                _short_join(row.modalities, limit=28),
                _format_status(row.integration_status),
                _format_status(row.verification_status),
                _short_join(row.metrics, limit=34),
                key=row.benchmark_id,
            )

    def _update_selection_summary(self, visible_models: int, visible_benchmarks: int) -> None:
        """Update the selection-summary ``Static`` widget with the current model or benchmark selection."""
        try:
            summary = self.query_one("#selection-summary", Static)
        except NoMatches:
            return
        model = self.selected_model_id or "-"
        if self.action == "eval":
            bench = self.selected_benchmark_id or "-"
            variant = ""
            if self.eval_intent == "score-artifacts":
                catalog = f"{visible_benchmarks} benchmarks"
                body = f"[bold cyan]Artifact directory[/]  [dim]→[/]  [bold magenta]{bench}[/]"
            else:
                catalog = f"{visible_models} models · {visible_benchmarks} benchmarks"
                body = f"[bold cyan]{model}[/]  [dim]→[/]  [bold magenta]{bench}[/]"
        elif self.action == "infer":
            variant = self._selected_ckpt_type_label()
            variant_part = f"  [dim]·[/]  [bold yellow]{variant}[/]" if variant else ""
            catalog = f"{visible_models} models"
            body = f"[bold cyan]{model}[/]{variant_part}"
        else:
            variant = ""
            catalog = f"{visible_models} models"
            body = "[bold green]Local Studio server[/]"

        action_label = icons.action_labels().get(self.action, self.action)
        summary.update(
            f"[bold]{action_label}[/]\n"
            f"{body}\n"
            f"[dim]{catalog}[/]"
        )

    def _update_pane_titles(self, visible_models: int, visible_benchmarks: int) -> None:
        """Update the pane-title labels to reflect current counts and action mode."""
        try:
            self.query_one("#models-pane-title", Label).update(icons.models_title(visible_models))
            if self.action == "eval":
                self.query_one("#benchmarks-pane-title", Label).update(icons.benchmarks_title(visible_benchmarks))
        except NoMatches:
            return


    def _update_command(self) -> None:
        """Refresh the command preview, conda status, selection status, and selection summary widgets."""
        try:
            command_text = self._command_text()
            command_widget = self.query_one("#command", Static)
            if command_text.startswith("Select ") or command_text.startswith("Ckpt override"):
                command_widget.update(Text(command_text, style="yellow"))
            else:
                command_widget.update(Syntax(command_text, "bash", theme="monokai", word_wrap=True))
            
            self.query_one("#conda-status", Static).update(self._conda_status_text())

            if self.action == "eval":
                eval_source = "artifacts" if self.eval_intent == "score-artifacts" else self.selected_model_id or "-"
                self.query_one("#selection-status", Static).update(
                    icons.status_eval(eval_source, self.selected_benchmark_id or "-")
                )
            elif self.action == "infer":
                ckpt_type = self._selected_ckpt_type_label()
                self.query_one("#selection-status", Static).update(
                    icons.status_infer(self.selected_model_id or "-", ckpt_type or "-")
                )
            else:
                self.query_one("#selection-status", Static).update(icons.status_studio())
            filter_value = self.query_one("#filter", Input).value
            models = [
                row for row in self.model_rows if self._model_visible_for_action(row) and self._matches_model(row, filter_value)
            ]
            benchmarks = [row for row in self.benchmark_rows if self._matches_benchmark(row, filter_value)]
            self._update_selection_summary(len(models), len(benchmarks))
        except NoMatches:
            return

    def _command(self) -> tuple[str, ...]:
        """Build the shell command tuple for the current action and selections.

        Returns:
            A tuple of strings suitable for ``asyncio.create_subprocess_exec``.

        Raises:
            ValueError: When an invalid checkpoint type or media-file override is detected.
        """
        if self.action == "infer":
            controls = set(infer_control_fields(self.selected_model_id, self.ckpt_type))

            def infer_input_value(field: str, widget_id: str) -> str | None:
                if field not in controls:
                    return None
                return _blank_to_none(self.query_one(f"#{widget_id}", Input).value)

            raw_ckpt_path = _blank_to_none(self.query_one("#ckpt-path", Input).value)
            input_path, video_path, ckpt_path = self._normalized_infer_paths(
                infer_input_value("input", "input-path"),
                infer_input_value("video", "video-path"),
                raw_ckpt_path,
            )
            if input_path is not None and "input" not in controls:
                input_path = None
            if video_path is not None and "video" not in controls:
                video_path = None
            if (
                ckpt_path is None
                and raw_ckpt_path
                and _looks_like_media_file(raw_ckpt_path)
                and input_path is None
                and video_path is None
            ):
                raise ValueError("Media files are only valid for models with visible Input or Video controls.")
            cam_type = _normalize_select_value(self.query_one("#cam-type", Select).value) if "cam_type" in controls else None
            return build_model_infer_command(
                model_id=self.selected_model_id,
                output_dir=Path(self.query_one("#output-dir", Input).value or self._default_output_dir()),
                ckpt_type=self.ckpt_type,
                ckpt_root=_blank_to_none(self.query_one("#ckpt-root", Input).value),
                ckpt_path=ckpt_path,
                input_path=input_path,
                input_dir=infer_input_value("input_dir", "input-dir"),
                video_path=video_path,
                trajectory_file=infer_input_value("trajectory_file", "trajectory-file"),
                prompt=infer_input_value("prompt", "prompt"),
                negative_prompt=infer_input_value("negative_prompt", "negative-prompt"),
                task=infer_input_value("task", "task"),
                mode=infer_input_value("mode", "mode"),
                resize_mode=infer_input_value("resize_mode", "resize-mode"),
                size=infer_input_value("size", "size"),
                frames=infer_input_value("frames", "frames"),
                steps=infer_input_value("steps", "steps"),
                frames_per_generation=infer_input_value("frames_per_generation", "frames-per-generation"),
                guidance_scale=infer_input_value("guidance_scale", "guidance-scale"),
                seed=infer_input_value("seed", "seed"),
                fps=infer_input_value("fps", "fps"),
                dtype=infer_input_value("dtype", "dtype"),
                max_sequence_length=infer_input_value("max_sequence_length", "max-sequence-length"),
                cam_type=cam_type,
                interactions=infer_input_value("interactions", "interactions"),
                output_formats=infer_input_value("output_formats", "output-formats"),
                trajectory=infer_input_value("trajectory", "trajectory"),
                angle=infer_input_value("angle", "angle"),
                distance=infer_input_value("distance", "distance"),
                orbit_radius=infer_input_value("orbit_radius", "orbit-radius"),
                zoom_ratio=infer_input_value("zoom_ratio", "zoom-ratio"),
                alpha_threshold=infer_input_value("alpha_threshold", "alpha-threshold"),
                static_scene=infer_input_value("static_scene", "static-scene"),
                low_vram=infer_input_value("low_vram", "low-vram"),
                disable_lora=infer_input_value("disable_lora", "disable-lora"),
                vis_rendering=infer_input_value("vis_rendering", "vis-rendering"),
                offload_t5=infer_input_value("offload_t5", "offload-t5"),
                offload_transformer_during_vae=infer_input_value(
                    "offload_transformer_during_vae",
                    "offload-transformer-during-vae",
                ),
                offload_vae=infer_input_value("offload_vae", "offload-vae"),
                output_path=infer_input_value("output_path", "output-path"),
                conda_envs_root=_blank_to_none(self.query_one("#conda-envs-root", Input).value),
                gpu=_blank_to_none(self.query_one("#gpu", Input).value),
            )
        if self.action == "ui":
            return build_studio_command()
        return build_model_benchmark_command(
            model_id=self.selected_model_id,
            benchmark_id=self.selected_benchmark_id,
            output_dir=self.query_one("#output-dir", Input).value or self._default_output_dir(),
            model_manifest_dir=self.model_manifest_dir,
            benchmark_manifest_dir=self.benchmark_manifest_dir,
            generated_artifact_dir=(
                _blank_to_none(self.query_one("#eval-artifact-dir", Input).value)
                if self.eval_intent == "score-artifacts"
                else None
            ),
            metrics=_split_csv_values([self.query_one("#metrics", Input).value]),
        )

    def _command_text(self) -> str:
        """Return a human-readable shell command string, or a selection prompt if IDs are missing."""
        model_required = self.action != "eval" or self.eval_intent == "model-benchmark"
        if self.action != "ui" and model_required and not self.selected_model_id:
            return "Select a model."
        if self.action == "eval" and not self.selected_benchmark_id:
            return "Select a benchmark."
        if self.action == "eval" and self.eval_intent == "score-artifacts" and not self.eval_artifact_dir:
            return "Select an artifact directory."
        try:
            return format_shell_command(self._command())
        except ValueError as exc:
            return str(exc)

    def _conda_status_text(self) -> str:
        """Build a Rich-markup string summarizing the Conda environment availability."""
        status = dict(self.catalog.conda_status or {})
        available = "yes" if status.get("conda_available") else "no"
        return (
            f"Conda [green]available[/]=[bold]{available}[/]\n"
            f"[dim]active[/]=[bold]{status.get('active_env') or '-'}[/] "
            f"[dim]envs[/]=[bold]{status.get('envs_existing', 0)}/{status.get('env_specs', 0)}[/]\n"
            f"[dim]root[/]={status.get('env_root') or '-'}"
        )

    def _current_output_dir(self) -> Path:
        """Return the current output directory path, expanded from the widget value or default."""
        try:
            raw = self.query_one("#output-dir", Input).value
        except NoMatches:
            raw = ""
        return Path(raw or self._default_output_dir()).expanduser()

    def _preflight_lines(self) -> tuple[str, ...]:
        """Collect a tuple of preflight status lines for the current action."""
        lines: list[str] = []
        output_dir = self._current_output_dir()
        lines.append(_path_status("output", output_dir, require_exists=False))
        if self.action == "infer":
            lines.extend(self._infer_preflight_lines())
        elif self.action == "eval":
            if self.eval_intent == "score-artifacts":
                artifact_dir = _blank_to_none(self.query_one("#eval-artifact-dir", Input).value)
                lines.append(_path_status("artifacts", Path(artifact_dir).expanduser() if artifact_dir else None))
                lines.append("  metrics: complete benchmark protocol")
            else:
                metrics = _split_csv_values([self.query_one("#metrics", Input).value])
                lines.append(f"  metrics: {', '.join(metrics) if metrics else 'default'}")
        else:
            lines.append("  studio: launches the local Studio server")
        return tuple(lines)

    def _infer_preflight_lines(self) -> tuple[str, ...]:
        """Collect preflight status lines specific to inference mode."""
        lines: list[str] = []
        controls = set(infer_control_fields(self.selected_model_id, self.ckpt_type))
        selected_row = next((row for row in self.model_rows if row.model_id == self.selected_model_id), None)
        studio_runtime = bool(selected_row and "studio_runtime" in selected_row.notes)
        ckpt_root = _blank_to_none(self.query_one("#ckpt-root", Input).value)
        conda_root = _blank_to_none(self.query_one("#conda-envs-root", Input).value)
        ckpt_path = _blank_to_none(self.query_one("#ckpt-path", Input).value)
        input_path = _blank_to_none(self.query_one("#input-path", Input).value)
        input_dir = _blank_to_none(self.query_one("#input-dir", Input).value)
        video_path = _blank_to_none(self.query_one("#video-path", Input).value)
        trajectory_file = _blank_to_none(self.query_one("#trajectory-file", Input).value)
        if not studio_runtime:
            lines.append(_path_status("ckpt root", Path(ckpt_root).expanduser() if ckpt_root else None))
            lines.append(_path_status("env root", Path(conda_root).expanduser() if conda_root else None))
        if ckpt_path:
            if _looks_like_remote_id(ckpt_path):
                lines.append(f"  model ref override: remote/id {ckpt_path}")
            else:
                lines.append(_path_status("model ref override", Path(ckpt_path).expanduser()))
        if "input" in controls:
            lines.append(_path_status("input", Path(input_path).expanduser() if input_path else None))
        if "input_dir" in controls:
            lines.append(_path_status("input dir", Path(input_dir).expanduser() if input_dir else None))
        if "video" in controls:
            lines.append(_path_status("video", Path(video_path).expanduser() if video_path else None))
        if "trajectory_file" in controls:
            lines.append(_path_status("trajectory file", Path(trajectory_file).expanduser() if trajectory_file else None))
        gpu = _blank_to_none(self.query_one("#gpu", Input).value)
        lines.append(f"  gpu: {gpu or 'runtime default'}")
        return tuple(lines)

    def _default_output_dir(self) -> Path:
        """Compute the default output directory path based on the current action and selection IDs."""
        if self.action == "infer":
            if not self.selected_model_id:
                return self.base_output_dir / "infer"
            return self.base_output_dir / "infer"
        if self.action == "ui":
            return self.base_output_dir / "ui"
        if self.eval_intent == "score-artifacts" and self.selected_benchmark_id:
            return self.base_output_dir / f"artifacts__{self.selected_benchmark_id}"
        if not self.selected_model_id or not self.selected_benchmark_id:
            return self.base_output_dir
        return self.base_output_dir / f"{self.selected_model_id}__{self.selected_benchmark_id}"

    def _start_selected_command(self) -> None:
        """Launch the selected command as an async subprocess, logging output to the RichLog."""
        if self.running_task is not None and not self.running_task.done():
            self._write_log("A run is already active. Stop it before starting another run.")
            return
        try:
            command = self._command()
        except ValueError as exc:
            self._write_log(str(exc))
            return
        self.stop_requested = False
        self.running_task = asyncio.create_task(self._run_command_task(command))
        self._sync_run_controls()

    async def _run_command_task(self, command: tuple[str, ...]) -> None:
        """Execute *command* as an async subprocess, streaming output to the RichLog panel."""
        self._write_log("$ " + format_shell_command(command))
        process: asyncio.subprocess.Process | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            self.running_process = process
            self._write_log("Run started.")
            assert process.stdout is not None
            async for raw_line in process.stdout:
                self._write_log(raw_line.decode(errors="replace").rstrip())
                await asyncio.sleep(0)
            exit_code = await process.wait()
            message = "Run stopped" if self.stop_requested else "Run exited"
            self._write_log(f"{message} with code {exit_code}.")
        except asyncio.CancelledError:
            if process is not None and process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            self._write_log("Run cancelled.")
            raise
        except OSError as exc:
            self._write_log(f"Unable to start command: {type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - report unexpected runner failures without crashing the TUI.
            self._write_log(f"Run failed: {type(exc).__name__}: {exc}")
        finally:
            self.running_process = None
            self.running_task = None
            self.stop_requested = False
            self._sync_run_controls()

    async def _stop_process(self) -> None:
        """Terminate the running subprocess gracefully, with a 5s kill fallback."""
        if self.running_task is None or self.running_task.done():
            self._write_log("No active run.")
            return
        self.stop_requested = True
        if self.running_process is None:
            self.running_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.running_task
            return
        self._write_log("Stopping active run...")
        self.running_process.terminate()
        try:
            await asyncio.wait_for(self.running_process.wait(), timeout=5)
        except asyncio.TimeoutError:
            self.running_process.kill()
            await self.running_process.wait()

    def _write_log(self, message: str) -> None:
        """Append *message* to the RichLog execution panel."""
        try:
            self.query_one("#logs", RichLog).write(message)
        except Exception:
            return

    def _sync_run_controls(self) -> None:
        """Enable/disable the Run and Stop buttons based on whether a process is active."""
        running = self.running_task is not None and not self.running_task.done()
        try:
            self.query_one("#run", Button).disabled = running
            self.query_one("#stop", Button).disabled = not running
        except Exception:
            return

    def _sync_command_preview_layout(self) -> None:
        """Toggle the command-preview collapsible and update its button label."""
        try:
            preview = self.query_one("#command-preview", Collapsible)
            button = self.query_one("#toggle-command", Button)
        except NoMatches:
            return
        preview.collapsed = not self.command_preview_visible
        button.label = " Hide" if self.command_preview_visible else " Cmd"
        self._sync_run_builder_height()

    def _sync_infer_control_layout(self) -> None:
        """Show / hide per-model inference fields based on the active variant's control spec."""
        try:
            control_specs = (
                {spec.field_id: spec for spec in infer_control_specs(self.selected_model_id, self.ckpt_type)}
                if self.action == "infer"
                else {}
            )
            controls = set(control_specs)
            selected_row = next((row for row in self.model_rows if row.model_id == self.selected_model_id), None)
            studio_runtime = bool(selected_row and "studio_runtime" in selected_row.notes)
            script_infer = self.action == "infer" and not studio_runtime
            self._sync_infer_field_text(control_specs)
            self._set_field_visible("prompt", "prompt-label", "prompt", "prompt" in controls)
            self._set_field_visible(
                "negative_prompt",
                "negative-prompt-label",
                "negative-prompt",
                "negative_prompt" in controls,
            )
            self._set_field_visible("input", "input-path-label", "input-path", "input" in controls)
            self._set_field_visible("input_dir", "input-dir-label", "input-dir", "input_dir" in controls)
            self._set_field_visible("video", "video-path-label", "video-path", "video" in controls)
            self._set_field_visible("task", "task-label", "task", "task" in controls)
            self._set_field_visible("mode", "mode-label", "mode", "mode" in controls)
            self._set_field_visible("interactions", "interactions-label", "interactions", "interactions" in controls)
            self._set_field_visible(
                "output_formats",
                "output-formats-label",
                "output-formats",
                "output_formats" in controls,
            )
            self._set_field_visible("trajectory", "trajectory-label", "trajectory", "trajectory" in controls)
            for field in ("size", "frames", "steps", "frames_per_generation", "seed", "fps"):
                input_id = "frames-per-generation" if field == "frames_per_generation" else field
                if field == "frames_per_generation":
                    self._set_field_visible(field, "frames-per-generation-label", input_id, field in controls)
                else:
                    self._set_field_visible(field, f"{field}-label", input_id, field in controls)
            self._set_select_visible("cam-type-label", "cam-type", "cam_type" in controls)
            self._set_field_visible("guidance_scale", "guidance-scale-label", "guidance-scale", "guidance_scale" in controls)
            self._set_field_visible("dtype", "dtype-label", "dtype", "dtype" in controls)
            self._set_field_visible(
                "max_sequence_length",
                "max-sequence-length-label",
                "max-sequence-length",
                "max_sequence_length" in controls,
            )
            self._set_field_visible("output_path", "output-path-label", "output-path", "output_path" in controls)
            self._set_field_visible(
                "trajectory_file",
                "trajectory-file-label",
                "trajectory-file",
                "trajectory_file" in controls,
            )
            self._set_field_visible("resize_mode", "resize-mode-label", "resize-mode", "resize_mode" in controls)
            self._set_field_visible("angle", "angle-label", "angle", "angle" in controls)
            self._set_field_visible("distance", "distance-label", "distance", "distance" in controls)
            self._set_field_visible(
                "orbit_radius",
                "orbit-radius-label",
                "orbit-radius",
                "orbit_radius" in controls,
            )
            self._set_field_visible("zoom_ratio", "zoom-ratio-label", "zoom-ratio", "zoom_ratio" in controls)
            self._set_field_visible(
                "alpha_threshold",
                "alpha-threshold-label",
                "alpha-threshold",
                "alpha_threshold" in controls,
            )
            self._set_field_visible("static_scene", "static-scene-label", "static-scene", "static_scene" in controls)
            self._set_field_visible("low_vram", "low-vram-label", "low-vram", "low_vram" in controls)
            self._set_field_visible("disable_lora", "disable-lora-label", "disable-lora", "disable_lora" in controls)
            self._set_field_visible(
                "vis_rendering",
                "vis-rendering-label",
                "vis-rendering",
                "vis_rendering" in controls,
            )
            self._set_field_visible("offload_t5", "offload-t5-label", "offload-t5", "offload_t5" in controls)
            self._set_field_visible(
                "offload_transformer_during_vae",
                "offload-transformer-during-vae-label",
                "offload-transformer-during-vae",
                "offload_transformer_during_vae" in controls,
            )
            self._set_field_visible("offload_vae", "offload-vae-label", "offload-vae", "offload_vae" in controls)

            self._set_group_visible("prompt-controls", "prompt" in controls)
            self._set_group_visible("negative-prompt-controls", "negative_prompt" in controls)
            self._set_group_visible("media-controls", bool({"input", "video"} & controls))
            self._set_group_visible("flashworld-controls", bool({"input_dir", "output_formats"} & controls))
            self._set_group_visible("task-controls", bool({"task", "mode"} & controls))
            self._set_group_visible("eval-controls", self.action == "eval")
            self._set_group_visible("path-controls", self.action in {"infer", "eval"})
            self._set_field_visible("ckpt_root", "ckpt-root-label", "ckpt-root", script_infer)
            self._set_field_visible("conda_envs_root", "conda-envs-root-label", "conda-envs-root", script_infer)
            self._set_field_visible("gpu", "gpu-label", "gpu", self.action == "infer")
            self._set_group_visible("runtime-controls", script_infer)
            self._set_group_visible("interaction-controls", bool({"interactions", "trajectory"} & controls))
            self._set_group_visible("trajectory-file-controls", bool({"trajectory_file", "resize_mode"} & controls))
            self._set_group_visible(
                "sampling-controls",
                bool({"size", "frames", "steps", "seed"} & controls),
            )
            self._set_group_visible(
                "sampling-controls-b",
                bool({"fps", "guidance_scale", "dtype", "max_sequence_length"} & controls),
            )
            self._set_group_visible("astra-controls", bool({"cam_type", "frames_per_generation"} & controls))
            self._set_group_visible(
                "camera-controls",
                bool({"angle", "distance", "orbit_radius", "zoom_ratio"} & controls),
            )
            self._set_group_visible(
                "neoverse-controls",
                bool({"alpha_threshold", "static_scene"} & controls),
            )
            self._set_group_visible(
                "neoverse-controls-b",
                bool({"low_vram", "disable_lora", "vis_rendering"} & controls),
            )
            self._set_group_visible(
                "flashworld-offload-controls",
                bool({"offload_t5", "offload_transformer_during_vae", "offload_vae"} & controls),
            )
            self._set_group_visible("checkpoint-controls", self.action == "infer")
            self._set_group_visible("checkpoint-controls-b", self.action == "infer")

            general_prompt_groups = (
                "prompt-controls",
                "negative-prompt-controls",
                "media-controls",
                "flashworld-controls",
                "task-controls",
                "eval-controls",
            )
            general_path_groups = ("path-controls", "runtime-controls")
            self._sync_section_visible(
                "section-prompt-title",
                "section-prompt-rule",
                general_prompt_groups,
            )
            self._sync_section_visible(
                "section-paths-title",
                "section-paths-rule",
                general_path_groups,
            )
            self._sync_section_visible(
                "section-model-title",
                "section-model-rule",
                ("neoverse-controls", "neoverse-controls-b", "flashworld-offload-controls"),
            )

            general_visible = self.action in {"infer", "eval"}
            sampling_visible = bool(
                {"size", "frames", "steps", "seed", "fps", "guidance_scale", "dtype", "max_sequence_length"} & controls
            )
            camera_visible = bool(
                {
                    "interactions",
                    "trajectory",
                    "trajectory_file",
                    "resize_mode",
                    "cam_type",
                    "frames_per_generation",
                    "angle",
                    "distance",
                    "orbit_radius",
                    "zoom_ratio",
                }
                & controls
            )
            model_visible = self.action == "infer"
            self._set_tab_enabled("tab-general", general_visible)
            self._set_tab_enabled("tab-sampling", sampling_visible)
            self._set_tab_enabled("tab-camera", camera_visible)
            self._set_tab_enabled("tab-model", model_visible)
            self._sync_tab_empty_states(
                {
                    "tab-general": general_prompt_groups + general_path_groups,
                    "tab-sampling": ("sampling-controls", "sampling-controls-b"),
                    "tab-camera": (
                        "interaction-controls",
                        "trajectory-file-controls",
                        "astra-controls",
                        "camera-controls",
                    ),
                    "tab-model": (
                        "checkpoint-controls",
                        "checkpoint-controls-b",
                        "neoverse-controls",
                        "neoverse-controls-b",
                        "flashworld-offload-controls",
                    ),
                }
            )
            self.query_one("#config-tabs", TabbedContent).display = "none" if self.action == "ui" else "block"
            self.query_one("#studio-panel", Vertical).display = "block" if self.action == "ui" else "none"
            self._sync_active_tab()
            self._sync_run_builder_height()
        except NoMatches:
            return

    def _sync_infer_field_text(self, specs: dict[str, InferControlSpec]) -> None:
        """Update label text and input placeholders from variant-specific control specs."""
        for field_id, spec in specs.items():
            label_id, input_id = _control_widget_ids(field_id)
            if label_id:
                self._update_label(label_id, spec.label)
            if input_id and spec.placeholder:
                self._update_placeholder(input_id, spec.placeholder)

    def _update_label(self, label_id: str, text: str) -> None:
        """Update the text of a ``Label`` widget identified by *label_id*."""
        try:
            self.query_one(f"#{label_id}", Label).update(text)
        except NoMatches:
            return

    def _update_placeholder(self, input_id: str, placeholder: str) -> None:
        """Update the placeholder text of an ``Input`` widget identified by *input_id*."""
        try:
            self.query_one(f"#{input_id}", Input).placeholder = placeholder
        except NoMatches:
            return

    def _set_field_visible(self, _field: str, label_id: str, input_id: str, visible: bool) -> None:
        """Show or hide an input field and its label, propagating to the parent field-group."""
        try:
            label = self.query_one(f"#{label_id}", Label)
            input_widget = self.query_one(f"#{input_id}", Input)
        except NoMatches:
            return
        label.display = visible
        input_widget.display = visible
        if label.parent and label.parent.has_class("field-group"):
            label.parent.display = "block" if visible else "none"

    def _set_select_visible(self, label_id: str, select_id: str, visible: bool) -> None:
        """Show or hide a ``Select`` widget and its label."""
        try:
            label = self.query_one(f"#{label_id}", Label)
            select_widget = self.query_one(f"#{select_id}", Select)
        except NoMatches:
            return
        label.display = visible
        select_widget.display = visible
        if label.parent and label.parent.has_class("field-group"):
            label.parent.display = "block" if visible else "none"

    def _set_group_visible(self, group_id: str, visible: bool) -> None:
        """Show or hide a container widget group identified by *group_id*."""
        widget = self.query_one(f"#{group_id}")
        widget.display = "block" if visible else "none"

    def _group_is_visible(self, group_id: str) -> bool:
        """Return whether the widget group identified by *group_id* is currently displayed."""
        try:
            return self.query_one(f"#{group_id}").display != "none"
        except NoMatches:
            return False

    def _sync_section_visible(self, title_id: str, rule_id: str, group_ids: tuple[str, ...]) -> None:
        """Show or hide a section title and rule based on whether any child group is visible."""
        visible = any(self._group_is_visible(group_id) for group_id in group_ids)
        self.query_one(f"#{title_id}", Static).display = "block" if visible else "none"
        self.query_one(f"#{rule_id}", Rule).display = "block" if visible else "none"

    def _sync_tab_empty_states(self, tab_groups: dict[str, tuple[str, ...]]) -> None:
        """Toggle the "no options" empty-state labels per tab based on child group visibility."""
        for tab_id, group_ids in tab_groups.items():
            empty_id = f"{tab_id}-empty"
            try:
                empty = self.query_one(f"#{empty_id}", Static)
            except NoMatches:
                continue
            empty.display = "none" if any(self._group_is_visible(group_id) for group_id in group_ids) else "block"

    def _set_tab_enabled(self, tab_id: str, enabled: bool) -> None:
        """Enable or disable a ``TabPane`` identified by *tab_id*."""
        self.query_one(f"#{tab_id}", TabPane).disabled = not enabled

    def _sync_active_tab(self) -> None:
        """Switch to the first non-disabled tab if the currently active tab is disabled."""
        tabs = self.query_one("#config-tabs", TabbedContent)
        order = ("tab-general", "tab-sampling", "tab-camera", "tab-model")
        active = tabs.active
        if active in order and not self.query_one(f"#{active}", TabPane).disabled:
            return
        for tab_id in order:
            if not self.query_one(f"#{tab_id}", TabPane).disabled:
                tabs.active = tab_id
                return

    def _set_tab_visible(self, tab_id: str, visible: bool) -> None:
        """Alias for :meth:`_set_tab_enabled` — show or hide a tab by its ID."""
        self._set_tab_enabled(tab_id, visible)

    def _sync_run_builder_height(self) -> None:
        """Force a layout refresh on the run-builder container to accommodate visibility changes."""
        try:
            self.query_one("#run-builder", Vertical).refresh(layout=True)
        except NoMatches:
            return

    def _brand_logo_width(self) -> int:
        """Compute an appropriate Braille-art logo width constrained by terminal size and aspect ratio."""

        # NOTE: The cropped logo's bounding box is roughly 921×713 pixels.
        aspect = 921 / 713

        # Determine maximum acceptable height in rows — never so large that
        # the CSS/Textual layout engine gets confused.
        max_rows = max(18, int(self.size.height * 0.35))

        # A width of ~60 chars yields ~23 rows of Braille art.
        max_width = int(max_rows * 2 * aspect)

        # NOTE: Render width is constrained by available terminal width and our max height allowance.
        available = max(40, int(self.size.width * 0.4) - 8)
        render_width = max(40, min(available, max_width, 80))
        return render_width

    def _sync_brand_logo(self) -> None:
        """Re-render the brand logo at the correct width, using a short timer to debounce layout changes."""
        try:
            logo = self.query_one("#brand-logo", Static)
        except NoMatches:
            return
        
        # NOTE: Cancel any pending logo update before scheduling a new one
        if hasattr(self, "_logo_update_timer"):
            self._logo_update_timer.stop()
            
        def do_update():
            # NOTE: For round borders, wait for size layout to settle before rendering
            try:
                logo.update(render_brand_logo(width_chars=self._brand_logo_width()))
            except Exception:
                pass
            
        self._logo_update_timer = self.set_timer(0.05, do_update)

    def _sync_action_theme(self) -> None:
        """Apply the action-specific Textual theme and CSS mode class."""
        self.theme = themes.get(self.action, "textual-dark")
        try:
            status_pane = self.query_one("#status-pane")
            # ── Remove old mode classes, add new one ──
            status_pane.remove_class("mode-infer", "mode-eval", "mode-ui")
            status_pane.add_class(f"mode-{self.action}")
            
            # NOTE: Update the status-pane border title dynamically
            mode_names = {"infer": "Inference", "eval": "Evaluation", "ui": "Studio"}
            status_pane.border_title = f" OpenEnvision / {mode_names.get(self.action, '')} "
        except NoMatches:
            return

    def _sync_action_layout(self) -> None:
        """Toggle the benchmarks pane visibility and update filter placeholder based on action."""
        show_benchmarks = self.action == "eval"
        score_artifacts = show_benchmarks and self.eval_intent == "score-artifacts"
        try:
            self.query_one("#benchmarks-pane", Vertical).display = show_benchmarks
            self.query_one("#models-pane", Vertical).display = not score_artifacts
            filter_input = self.query_one("#filter", Input)
        except NoMatches:
            return
        if score_artifacts:
            filter_input.placeholder = "Filter benchmarks"
        else:
            filter_input.placeholder = "Filter models and benchmarks" if show_benchmarks else "Filter models"
        self._set_group_visible("eval-artifact-controls", score_artifacts)
        self._set_group_visible("eval-metrics-controls", show_benchmarks and not score_artifacts)
        self._sync_action_theme()
        self._sync_action_select()
        self._sync_infer_control_layout()

    def _sync_action_select(self) -> None:
        """Sync the action-mode ``Select`` widget value with the internal ``action`` attribute."""
        try:
            select = self.query_one("#action", Select)
        except NoMatches:
            return
        if self.action in TUI_ACTIONS:
            select.value = self.action

    def _model_visible_for_action(self, row: ModelCatalogRow) -> bool:
        """Return whether *row* should appear in the models table for the current action."""
        if self.action == "infer":
            return is_infer_model_row(row)
        if self.action == "eval":
            return row.runner_kind not in {"infer_script", "studio_runtime"}
        return True

    def _ckpt_type_options(self) -> tuple[tuple[str, str], ...]:
        """Return label/value pairs for the checkpoint type (variant) selector."""
        return infer_variant_options(self.selected_model_id) or (("default", self.selected_model_id),)

    def _sync_ckpt_type_for_selected_model(self) -> None:
        """Resolve and store a valid checkpoint type for the currently selected model family."""
        options = self._ckpt_type_options()
        values = {value for _label, value in options}
        if self.ckpt_type in values:
            return
        if self.ckpt_type:
            try:
                resolved = resolve_infer_model_variant(self.selected_model_id, self.ckpt_type)
            except ValueError:
                resolved = ""
            if resolved in values:
                self.ckpt_type = resolved
                return
        self.ckpt_type = options[0][1] if options else ""

    def _refresh_ckpt_type_select(self) -> None:
        """Refresh the checkpoint-type ``Select`` widget options and sync the stored value."""
        try:
            select = self.query_one("#ckpt-type", Select)
        except Exception:
            return
        options = self._ckpt_type_options()
        select.set_options(options)
        if options:
            select.value = self.ckpt_type
        select.disabled = self.action != "infer"

    def _selected_ckpt_type_label(self) -> str:
        """Return the display label for the current checkpoint type, or an empty string."""
        if not self.ckpt_type:
            return ""
        return infer_variant_label(self.ckpt_type)

    def _normalized_infer_paths(
        self,
        input_path: str | None,
        video_path: str | None,
        ckpt_path: str | None,
    ) -> tuple[str | None, str | None, str | None]:
        """Normalize inference path overrides, auto-routing media files to input or video slots.

        Raises:
            ValueError: When the checkpoint override is a media file that conflicts with
                explicit input or video overrides.
        """
        if not ckpt_path or not _looks_like_media_file(ckpt_path):
            return input_path, video_path, ckpt_path
        if input_path or video_path:
            raise ValueError(
                "Ckpt override points to an image/video file. Keep images in Input; "
                "Ckpt override must be a model directory or HuggingFace repo ID."
            )
        if _looks_like_video_file(ckpt_path):
            return input_path, ckpt_path, None
        return ckpt_path, video_path, None

    @staticmethod
    def _matches_model(row: ModelCatalogRow, query: str) -> bool:
        """Return whether *row* matches the case-insensitive filter *query*."""
        if not query:
            return True
        haystack = " ".join(
            [
                row.model_id,
                row.name,
                row.provider or "",
                row.source_status,
                row.integration_status,
                row.runner_status,
                row.runner_kind,
                row.runtime_profile or "",
                " ".join(row.runnable_variants),
                " ".join(infer_variant_label(variant) for variant in infer_model_variant_ids(row.model_id)),
                " ".join(row.tasks),
                " ".join(row.hf_repo_ids),
            ]
        ).casefold()
        return query in haystack

    @staticmethod
    def _matches_benchmark(row: BenchmarkCatalogRow, query: str) -> bool:
        """Return whether *row* matches the case-insensitive filter *query*."""
        if not query:
            return True
        haystack = " ".join(
            [
                row.benchmark_id,
                row.name,
                row.source_status,
                row.integration_status,
                row.verification_status,
                row.runner_target or "",
                row.install_profile or "",
                " ".join(row.domains),
                " ".join(row.modalities),
                " ".join(row.tags),
                " ".join(row.metrics),
                " ".join(row.hf_dataset_ids),
            ]
        ).casefold()
        return query in haystack


# ── Module-level helpers ──────────────────────────────────────────

# NOTE: Mapping from action mode to Textual theme name, applied when the action changes.
themes = {"infer": "textual-dark", "eval": "textual-dark", "ui": "textual-dark"}

def _normalize_select_value(value: object, *, allowed: Iterable[str] | None = None) -> str | None:
    """Normalize a ``Select`` widget value to a string, rejecting ``NULL`` and disallowed options."""
    if value is None or value is Select.NULL:
        return None
    text = str(value).strip()
    if not text or text.endswith(".NULL"):
        return None
    if allowed is not None and text not in set(allowed):
        return None
    return text


def _format_status(value: str) -> Text:
    """Colorize a status string using heuristic keyword matching (green/yellow/red)."""
    lowered = value.casefold()
    if any(token in lowered for token in ("ready", "integrated", "verified", "ok", "pass")):
        return Text(value, style="green")
    if any(token in lowered for token in ("partial", "warn", "pending")):
        return Text(value, style="yellow")
    if any(token in lowered for token in ("missing", "fail", "error", "blocked")):
        return Text(value, style="red")
    return Text(value)


def _short_join(values: Iterable[str], *, limit: int) -> str:
    """Join non-empty *values* with commas, truncating to *limit* characters."""
    text = ", ".join(str(value) for value in values if str(value).strip())
    if not text:
        return "-"
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _control_widget_ids(field_id: str) -> tuple[str | None, str | None]:
    """Map an inference control ``field_id`` to its ``(label_widget_id, input_widget_id)`` pair."""
    if field_id == "input":
        return "input-path-label", "input-path"
    if field_id == "input_dir":
        return "input-dir-label", "input-dir"
    if field_id == "video":
        return "video-path-label", "video-path"
    if field_id == "negative_prompt":
        return "negative-prompt-label", "negative-prompt"
    if field_id == "frames_per_generation":
        return "frames-per-generation-label", "frames-per-generation"
    if field_id == "guidance_scale":
        return "guidance-scale-label", "guidance-scale"
    if field_id == "max_sequence_length":
        return "max-sequence-length-label", "max-sequence-length"
    if field_id == "cam_type":
        return "cam-type-label", None
    if field_id == "output_formats":
        return "output-formats-label", "output-formats"
    if field_id == "trajectory_file":
        return "trajectory-file-label", "trajectory-file"
    if field_id == "resize_mode":
        return "resize-mode-label", "resize-mode"
    if field_id == "orbit_radius":
        return "orbit-radius-label", "orbit-radius"
    if field_id == "zoom_ratio":
        return "zoom-ratio-label", "zoom-ratio"
    if field_id == "alpha_threshold":
        return "alpha-threshold-label", "alpha-threshold"
    if field_id == "static_scene":
        return "static-scene-label", "static-scene"
    if field_id == "low_vram":
        return "low-vram-label", "low-vram"
    if field_id == "disable_lora":
        return "disable-lora-label", "disable-lora"
    if field_id == "vis_rendering":
        return "vis-rendering-label", "vis-rendering"
    if field_id == "offload_t5":
        return "offload-t5-label", "offload-t5"
    if field_id == "offload_transformer_during_vae":
        return "offload-transformer-during-vae-label", "offload-transformer-during-vae"
    if field_id == "offload_vae":
        return "offload-vae-label", "offload-vae"
    if field_id == "output_path":
        return "output-path-label", "output-path"
    widget_id = field_id.replace("_", "-")
    return f"{widget_id}-label", widget_id


def _clip_text(value: str, limit: int) -> str:
    """Truncate *value* to *limit* characters, appending ``…`` when truncated."""
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def _blank_to_none(value: str | None) -> str | None:
    """Convert a blank string to ``None``, stripping whitespace."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _split_csv_values(values: Iterable[str] | None) -> tuple[str, ...]:
    """Split repeatable / comma-separated CLI values into a flat tuple of individual items."""
    items: list[str] = []
    for value in values or ():
        for item in str(value).split(","):
            item = item.strip()
            if item:
                items.append(item)
    return tuple(items)


def _path_status(label: str, path: Path | None, *, require_exists: bool = True) -> str:
    """Format a preflight status line for *path*, reporting existence, kind, or missing status."""
    if path is None:
        return f"  {label}: not set"
    expanded = path.expanduser()
    if expanded.exists():
        kind = "dir" if expanded.is_dir() else "file"
        return f"  {label}: ok {kind} {expanded}"
    if require_exists:
        return f"  {label}: missing {expanded}"
    return f"  {label}: will write {expanded}"


def _looks_like_remote_id(value: str) -> bool:
    """Heuristic check for whether *value* looks like a remote model reference (HuggingFace repo ID or URL)."""
    path = Path(value).expanduser()
    if path.exists() or path.is_absolute() or value.startswith("."):
        return False
    if "://" in value:
        return True
    suffix = path.suffix.casefold()
    if suffix in {
        ".pt",
        ".pth",
        ".safetensors",
        ".ckpt",
        ".bin",
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
    }:
        return False
    return bool(value.strip())


def _format_file_artifact(path: Path, *, root: Path | None = None) -> str:
    """Format a file artifact entry with its relative path and human-readable size."""
    label = str(path.relative_to(root)) if root is not None else str(path)
    return f"{label} ({_human_bytes(path.stat().st_size)})"


def _human_bytes(size: int) -> str:
    """Convert a byte count to a human-readable string (e.g. ``"1.2GiB"``)."""
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024.0
    return f"{size}B"


def _looks_like_media_file(path: str) -> bool:
    """Return whether *path* has an image or video file extension."""
    suffix = Path(path).suffix.casefold()
    return suffix in {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
    }


def _looks_like_video_file(path: str) -> bool:
    """Return whether *path* has a video file extension."""
    return Path(path).suffix.casefold() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
