"""Event-centric inference entry point.

Supports two modes:
  1. Single config:  --config path/to/config.yaml
  2. Batch mode:     --configs-list path/to/list.txt

In batch mode all heavyweight models are loaded once externally, then each
config gets a fresh pipeline instance that reuses those shared models.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from worldfoundry.synthesis.visual_generation.liveworld.liveworld_runtime.liveworld.pipelines.monitor_centric import (
    MonitorCentricEvolutionPipeline,
    load_monitor_centric_config,
)
from worldfoundry.synthesis.visual_generation.liveworld.liveworld_runtime.liveworld.pipelines.monitor_centric.shared_models import (
    create_shared_models,
)
from worldfoundry.synthesis.visual_generation.liveworld.liveworld_runtime.liveworld.utils import (
    set_seed,
)


def _prepare_config(config_path: str, output_root: str, device: str | None, system_config: str | None = None) -> dict:
    """Load config, set output dir and optional device override."""
    config_path = os.path.abspath(config_path)
    system_config_abs = os.path.abspath(system_config) if system_config else None
    config = load_monitor_centric_config(config_path, system_config_path=system_config_abs)

    combo_id = os.path.splitext(os.path.basename(config_path))[0]
    image_stem = os.path.basename(os.path.dirname(os.path.dirname(config_path)))
    output_dir = os.path.join(os.path.abspath(output_root), image_stem, combo_id)
    config.setdefault("output", {})["root"] = output_dir

    if device:
        config["runtime"]["device"] = device

    return config


def _inject_entity_names(config: dict, entity_names_str: str | None) -> None:
    """Inject CLI entity names into config for the pipeline to pick up."""
    if not entity_names_str:
        return
    names = [n.strip() for n in entity_names_str.split(",") if n.strip()]
    if names:
        config.setdefault("event", {})["cli_entity_names"] = names


def _run_single(config: dict) -> None:
    """Build pipeline from scratch and run once (single-config mode)."""
    output_dir = config.get("output", {}).get("root", "")
    if output_dir and os.path.isfile(os.path.join(output_dir, "final_video.mp4")):
        print(f"[skip] final_video.mp4 already exists in {output_dir}")
        return

    set_seed(config["runtime"]["seed"])
    pipeline = MonitorCentricEvolutionPipeline(config=config)

    iter_inputs = config["iter_input"]
    pipeline.run(iter_inputs)


def _run_batch(
    config_paths: list[str],
    output_root: str,
    device: str | None,
    entity_names_str: str | None = None,
    system_config: str | None = None,
) -> None:
    """Load all models once, then run each config with a fresh pipeline."""
    if not config_paths:
        print("[batch] No configs to run")
        return

    total = len(config_paths)
    done = 0

    # Load all heavyweight models once from the first config.
    first_config = _prepare_config(config_paths[0], output_root, device, system_config)
    _inject_entity_names(first_config, entity_names_str)
    print("[batch] Loading shared models...")
    shared = create_shared_models(first_config)

    for idx, cfg_path in enumerate(config_paths):
        combo_id = os.path.splitext(os.path.basename(cfg_path))[0]
        image_stem = os.path.basename(os.path.dirname(os.path.dirname(cfg_path)))
        label = f"{image_stem}/{combo_id}"

        # Skip if final_video.mp4 already exists.
        output_dir = os.path.join(os.path.abspath(output_root), image_stem, combo_id)
        if os.path.isfile(os.path.join(output_dir, "final_video.mp4")):
            print(f"\n[batch] ({idx + 1}/{total}) {label} — skip (final_video.mp4 exists)")
            done += 1
            continue

        print(f"\n[batch] ({idx + 1}/{total}) {label}")

        config = _prepare_config(cfg_path, output_root, device, system_config)
        _inject_entity_names(config, entity_names_str)
        set_seed(config["runtime"]["seed"])

        # Reset Stream3R session state (keep model loaded).
        if hasattr(shared.pointcloud_handler, "reset_session"):
            shared.pointcloud_handler.reset_session()

        # Fresh pipeline per config, all models shared.
        pipeline = MonitorCentricEvolutionPipeline(config=config, shared_models=shared)
        pipeline.run(config["iter_input"])
        done += 1
    print(f"\n[batch] done: {done}/{total} succeeded")


def main() -> None:
    parser = argparse.ArgumentParser(description="Event-centric inference entry")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="Path to a single event-centric YAML config")
    group.add_argument("--configs-list", help="Path to file listing config paths (one per line)")
    parser.add_argument(
        "--system-config", required=True, help="Path to system config YAML (model weights, runtime settings)"
    )
    parser.add_argument("--output-root", required=True, help="Output directory for inference results")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda:0")
    parser.add_argument(
        "--entity-names", default=None, help="Comma-separated entity names to always detect, e.g. 'person,dog'"
    )
    args = parser.parse_args()

    if args.configs_list:
        # Batch mode: read config paths from file, then remove the temp file.
        configs_list_path = Path(args.configs_list)
        with open(configs_list_path) as f:
            config_paths = [line.strip() for line in f if line.strip()]
        configs_list_path.unlink(missing_ok=True)
        _run_batch(config_paths, args.output_root, args.device, args.entity_names, args.system_config)
    else:
        # Single config mode.
        config = _prepare_config(args.config, args.output_root, args.device, args.system_config)
        _inject_entity_names(config, args.entity_names)
        _run_single(config)


if __name__ == "__main__":
    main()
