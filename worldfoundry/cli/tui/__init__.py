# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Terminal UI for browsing the catalog and building ``wf`` commands.

This package root stays import-light: ``wf --help`` registers the ``tui``
subcommand without loading Textual, which is imported only when the TUI runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def register_tui_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``wf tui`` with its inference, evaluation, and Studio flags."""
    parser = subparsers.add_parser(
        "tui",
        help="Browse models and launch runs in the terminal UI",
        description="Launch real WorldFoundry model inference, benchmark evaluation, and Studio commands.",
    )
    parser.add_argument("--model-manifest-dir", type=Path)
    parser.add_argument("--benchmark-manifest-dir", type=Path)
    parser.add_argument("--runtime-profile-dir", type=Path)
    parser.add_argument("--suite", action="append", dest="suite_ids", default=None)
    parser.add_argument("--model-id")
    parser.add_argument("--benchmark-id")
    parser.add_argument(
        "--metric", action="append", default=None, help="Benchmark metric id. Repeatable or comma-separated."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tui"))
    parser.add_argument("--input", type=Path, help="Optional input image/video passed to model inference scripts.")
    parser.add_argument(
        "--input-dir", "--input_dir", dest="input_dir", type=Path, help="FlashWorld JSON input directory/file."
    )
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
    parser.add_argument(
        "--frames-per-generation", "--chunk-frames", dest="frames_per_generation", help="Astra frame chunk size."
    )
    parser.add_argument("--guidance-scale", "--cfg-scale", dest="guidance_scale", help="Guidance/CFG scale override.")
    parser.add_argument("--seed", help="Seed override.")
    parser.add_argument("--fps", help="Output FPS override.")
    parser.add_argument("--dtype", help="Runtime dtype override when supported.")
    parser.add_argument(
        "--max-sequence-length", dest="max_sequence_length", help="Text encoder max sequence length override."
    )
    parser.add_argument("--cam-type", "--camera-type", dest="cam_type", help="Astra camera preset 1..7.")
    parser.add_argument(
        "--interactions", "--actions", dest="interactions", help="Comma-separated interaction/action list."
    )
    parser.add_argument(
        "--output-formats",
        "--output_formats",
        dest="output_formats",
        help="FlashWorld output formats, e.g. video,spz,ply.",
    )
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
    parser.add_argument(
        "--ckpt-type", "--weight-type", dest="ckpt_type", help="Checkpoint type inside the selected model family."
    )
    parser.add_argument("--ckpt-root", type=Path, help="Checkpoint root for model inference scripts.")
    parser.add_argument(
        "--ckpt-path", type=Path, help="Exact checkpoint/model path override for the selected infer model."
    )
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
    parser.set_defaults(func=handle_tui)


def handle_tui(args: argparse.Namespace) -> int:
    """Run the TUI action selected by the parsed ``wf tui`` arguments."""
    from .launcher import run_tui

    return run_tui(args)
