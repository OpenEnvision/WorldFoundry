# SPDX-FileCopyrightText: Copyright (c) 2026 ZJU-REAL/WorldMemBench contributors
# SPDX-License-Identifier: MIT

"""WorldMemBench command-line entry point."""

import argparse
import sys
from pathlib import Path

from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.data import load_samples, write_manifest
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.evaluate import evaluate
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.metrics import registry
from worldfoundry.evaluation.tasks.execution.runners.worldmembench.runtime.paths import DEFAULT_WEIGHTS_DIR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="worldmembench", description="WorldMemBench evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list-metrics", help="List available metrics")
    run = commands.add_parser("eval", help="Evaluate a video, video directory, or JSONL manifest")
    run.add_argument("input", help="Video file, directory, or JSONL manifest")
    run.add_argument("--metrics", nargs="+", help="Metrics to run (default: all available)")
    run.add_argument("--output", help="Result directory (default: eval_results/<timestamp>)")
    devices = run.add_mutually_exclusive_group()
    devices.add_argument("--device", help="Single device, e.g. cuda:0 or cpu (default: cuda)")
    devices.add_argument("--gpus", help="Comma-separated visible GPU indices, e.g. 0,1; one process per GPU")
    run.add_argument("--weights-dir", help=f"Weight directory (default: {DEFAULT_WEIGHTS_DIR})")
    run.add_argument("--config", help="YAML settings for checkpoints, metric batch sizes, and the DA3 runtime")
    vlm = run.add_argument_group("Dynamic Memory VLM")
    vlm.add_argument(
        "--vlm-backend", choices=("local", "api"), help="Self-hosted vLLM (default: local) or external API"
    )
    vlm.add_argument("--vlm-base-url", help="VLM endpoint (default: http://127.0.0.1:8001/v1)")
    vlm.add_argument("--vlm-model", help="Served model ID (default: Qwen3.8-27B)")
    vlm.add_argument(
        "--vlm-api-key-env",
        help="API key environment variable name, never the key itself (default: WORLDMEMBENCH_API_KEY)",
    )
    run.add_argument(
        "--expected-revisits",
        type=int,
        choices=(1, 2, 3),
        help="Fixed Human/Object revisit protocol, e.g. 2 for LRLR; omitted: preserve manifest K, otherwise detect returns automatically",
    )
    manifest = commands.add_parser("manifest", help="Create an input manifest without evaluating videos")
    manifest.add_argument("input", help="Video directory, file, or existing manifest")
    manifest.add_argument("--output", required=True, help="New output JSONL manifest")
    for command in (run, manifest):
        command.add_argument("--models", nargs="+", help="Filter generation model IDs")
        command.add_argument("--categories", nargs="+", help="Filter categories, e.g. human object environment")
        command.add_argument("--trajectories", nargs="+", help="Filter trajectory IDs")
        command.add_argument("--cases", nargs="+", help="Filter case IDs, e.g. case_002")
    args = parser.parse_args(argv)
    if args.command == "list-metrics":
        print("\n".join(registry.names()) or "WorldMemBench has no metrics yet.")
        return 0
    try:
        filters = dict(models=args.models, categories=args.categories, trajectories=args.trajectories, cases=args.cases)
        if args.command == "manifest":
            samples = load_samples(args.input, **filters)
            output = write_manifest(samples, args.output)
            print(f"Saved {len(samples)} samples to {output}")
            return 0
        config = {}
        if args.config:
            try:
                import yaml
            except ImportError as error:
                raise RuntimeError("--config requires PyYAML") from error
            try:
                config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as error:
                raise ValueError(f"invalid YAML config: {error}") from error
            if not isinstance(config, dict):
                raise ValueError("config must be a YAML mapping")
        vlm_options = {
            key: value
            for key, value in {
                "backend": args.vlm_backend,
                "base_url": args.vlm_base_url,
                "model": args.vlm_model,
                "api_key_env": args.vlm_api_key_env,
            }.items()
            if value is not None
        }
        if vlm_options:
            config.setdefault("metrics", {}).setdefault("dynamic_memory", {}).update(vlm_options)
        output, success = evaluate(
            args.input,
            metrics=args.metrics,
            output=args.output,
            expected_revisits=args.expected_revisits,
            **filters,
            device="cuda" if args.gpus is not None else args.device or config.get("device", "cuda"),
            gpus=args.gpus,
            weights_dir=args.weights_dir or config.get("cache_dir", DEFAULT_WEIGHTS_DIR),
            config=config,
            command=list(sys.orig_argv) if argv is None else ["worldmembench", *argv],
        )
    except (ValueError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Results saved to {output}")
    if not success:
        print("Some cases failed; see error fields in metrics/*/cases.jsonl.", file=sys.stderr)
    return 0 if success else 1
