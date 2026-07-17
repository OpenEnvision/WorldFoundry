"""CLI commands for inspecting model runners and managing base-model assets."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from worldfoundry.evaluation.utils import write_json

from .utils import json_dump

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT

# ── Models list and runtime runners ─────────────────────────────


def _handle_models_list(args: argparse.Namespace) -> int:
    """List available model types from the model registry."""
    from worldfoundry.evaluation.models.catalog.registry import discover_model_registry

    registry = discover_model_registry()
    items = [item.to_dict() for item in registry.list(args.family)]
    if args.json:
        json_dump(items)
        return 0

    for item in items:
        print(f"{item['model_type']} [{item['family']}] loader={item['has_loader']} infer={item['has_infer']}")
    return 0


def _handle_models_runtime_runners(args: argparse.Namespace) -> int:
    """Emit registered ``module:Class`` runner targets usable with ``worldfoundry-eval evaluate --mode model``.

    Parameters:
        args: CLI namespace; ``json`` selects JSON lines instead of plain text rows.
    """
    from worldfoundry.evaluation.models.runners.registry import model_runner_registry_report

    report = model_runner_registry_report()
    payload = [entry.to_dict() for entry in report.entries]
    if args.json:
        json_dump(payload)
        return 0
    for entry in payload:
        aliases = ", ".join(entry["aliases"]) if entry["aliases"] else "-"
        print(f"{entry['name']}: {entry['runner_target']} source={entry['source']} aliases={aliases}")
    for issue in report.issues:
        print(f"warning:{issue.code} {issue.name or '-'}: {issue.message}", file=sys.stderr)
    return 0


def _handle_models_visualizations(args: argparse.Namespace) -> int:
    """List official model visualizations supported by shared Studio backends."""

    from worldfoundry.studio.visualization.capability_registry import visualization_inventory

    inventory = visualization_inventory(family=args.family, model_id=args.model)
    if args.json:
        json_dump(inventory)
        return 0
    print(f"model visualizations: {inventory['model_count']}")
    for item in inventory["models"]:
        print(
            f"{item['model_id']} [{item['family']}] "
            f"renderers={','.join(item['renderers'])} backends={','.join(item['backends'])}"
        )
    return 0


# ── Base-model asset management ──────────────────────────────────


def _execute_download_commands(commands: list[list[str]]) -> list[dict[str, object]]:
    """Run a sequence of download commands and capture truncated stdout/stderr."""
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    executed = []
    for command in commands:
        completed = subprocess.run(
            [str(item) for item in command],
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
        executed.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            }
        )
    return executed


def _handle_models_assets(args: argparse.Namespace) -> int:
    """Plan or download reusable base-model checkpoints and data assets."""
    from worldfoundry.base_models.capabilities import base_model_inventory, base_model_materialization_plan

    if args.list:
        inventory = base_model_inventory()
        if args.report_path is not None:
            write_json(args.report_path, inventory)
        if args.json:
            json_dump(inventory)
        else:
            print(f"base-model capabilities: {inventory['capability_count']}")
            for item in inventory["capabilities"]:
                print(f"capability {item['id']} [{item['family']}]")
            print(f"base-model stacks: {inventory['stack_count']}")
            for item in inventory["stacks"]:
                print(f"stack {item['id']} [{item['family']}] -> {', '.join(item['capability_ids'])}")
            if args.report_path is not None:
                print("report:", args.report_path)
        return 0

    plan = base_model_materialization_plan(args.capability)
    executed = []
    if args.execute_downloads:
        executed = _execute_download_commands(plan.get("download_command_argvs", []))
        plan = base_model_materialization_plan(args.capability)
        plan["executed_downloads"] = executed

    if args.report_path is not None:
        write_json(args.report_path, plan)

    if args.json:
        json_dump(plan)
    else:
        print("base-model assets:", "ok" if plan["ok"] else "missing")
        if plan["stack_ids"]:
            print("stacks:", ", ".join(plan["stack_ids"]))
        print("capabilities:", ", ".join(plan["capability_ids"]))
        if plan["pip_install_packages"]:
            print("install:", "python -m pip install " + " ".join(plan["pip_install_packages"]))
        for command in plan["download_commands"]:
            print("download:", command)
        for command in plan["export_commands"]:
            print("env:", command)
        for action in plan["manual_actions"]:
            print("manual:", action)
        if args.report_path is not None:
            print("report:", args.report_path)
    if not args.execute_downloads:
        return 0
    execution_failed = any(int(item.get("returncode", 1)) != 0 for item in executed)
    return 0 if plan["ok"] and not execution_failed else 1


def _npz_value(path: Path, *keys: str):
    import numpy as np

    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    with np.load(path, allow_pickle=False) as payload:
        for key in keys:
            if key in payload:
                return payload[key].copy()
        if len(payload.files) == 1:
            return payload[payload.files[0]].copy()
    raise ValueError(f"{path} does not contain any of: {', '.join(keys)}")


def _media_frames(path: Path) -> list:
    import imageio.v3 as iio

    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        return [iio.imread(path)]
    return [frame for frame in iio.imiter(path)]


def _write_visualization(output: Path, frames: list, *, fps: float) -> None:
    from PIL import Image

    if len(frames) == 1:
        Image.fromarray(frames[0]).save(output)
        return
    import imageio.v2 as iio

    with iio.get_writer(output, fps=fps, codec="libx264") as writer:
        for frame in frames:
            writer.append_data(frame)


def _text_artifact(path: Path) -> str:
    import json

    if path.suffix.lower() != ".json":
        return path.read_text(encoding="utf-8").strip()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, str):
        return payload
    for key in ("caption", "text", "label", "action", "prediction"):
        if key in payload:
            return str(payload[key])
    raise ValueError("Text visualization JSON requires caption/text/label/action/prediction.")


def _handle_models_visualize(args: argparse.Namespace) -> int:
    """Render structured official perception output into a standard media artifact."""

    import numpy as np
    from PIL import Image

    from worldfoundry.studio.visualization.plugins.perception.render import (
        load_json_detections,
        render_depth,
        render_detections,
        render_feature_pca,
        render_keypoints,
        render_masks,
        render_normals,
        render_optical_flow,
        render_text_overlay,
        render_tracks,
    )

    artifact = args.artifact.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source = args.media.expanduser().resolve() if args.media else None

    if args.kind == "detection":
        if source is None:
            raise ValueError("detection visualization requires --media IMAGE")
        boxes, labels, scores = load_json_detections(artifact)
        rendered = render_detections(source, boxes, labels=labels, scores=scores, normalized=args.normalized)
        Image.fromarray(rendered).save(output)
        frame_count = 1
    elif args.kind == "mask":
        if source is None:
            raise ValueError("mask visualization requires --media IMAGE")
        rendered = render_masks(source, _npz_value(artifact, "masks", "mask", "segmentation"), alpha=args.alpha)
        Image.fromarray(rendered).save(output)
        frame_count = 1
    elif args.kind == "flow":
        rendered = render_optical_flow(_npz_value(artifact, "flow", "flows", "optical_flow"))
        Image.fromarray(rendered).save(output)
        frame_count = 1
    elif args.kind == "feature-pca":
        output_size = None
        if source is not None:
            with Image.open(source) as image:
                output_size = image.size
        rendered = render_feature_pca(
            _npz_value(artifact, "features", "feature", "patch_tokens"), output_size=output_size
        )
        Image.fromarray(rendered).save(output)
        frame_count = 1
    elif args.kind == "depth":
        values = _npz_value(artifact, "depth", "depths", "disparity")
        depth_frames = [values] if values.ndim == 2 else list(values)
        rendered_frames = [render_depth(frame, inverse=args.inverse) for frame in depth_frames]
        _write_visualization(output, rendered_frames, fps=args.fps)
        frame_count = len(rendered_frames)
    elif args.kind == "normal":
        values = _npz_value(artifact, "normals", "normal")
        normal_frames = [values] if values.ndim == 3 else list(values)
        rendered_frames = [render_normals(frame) for frame in normal_frames]
        _write_visualization(output, rendered_frames, fps=args.fps)
        frame_count = len(rendered_frames)
    elif args.kind == "keypoints":
        if source is None:
            raise ValueError("keypoints visualization requires --media IMAGE_OR_VIDEO")
        frames = _media_frames(source)
        with np.load(artifact, allow_pickle=False) as bundle:
            keypoints = bundle["keypoints"].copy()
            edges = bundle["edges"].copy() if "edges" in bundle else None
        keypoint_frames = [keypoints] if keypoints.ndim == 2 else list(keypoints)
        if len(frames) != len(keypoint_frames):
            raise ValueError("Keypoint and media frame counts must match.")
        rendered_frames = [
            render_keypoints(frame, points, edges=edges, normalized=args.normalized)
            for frame, points in zip(frames, keypoint_frames)
        ]
        _write_visualization(output, rendered_frames, fps=args.fps)
        frame_count = len(rendered_frames)
    elif args.kind == "text":
        if source is None:
            raise ValueError("text visualization requires --media IMAGE_OR_VIDEO")
        rendered_frames = [render_text_overlay(frame, _text_artifact(artifact)) for frame in _media_frames(source)]
        _write_visualization(output, rendered_frames, fps=args.fps)
        frame_count = len(rendered_frames)
    else:
        if source is None:
            raise ValueError("tracks visualization requires --media VIDEO")
        with np.load(artifact, allow_pickle=False) as bundle:
            tracks = bundle["tracks"].copy()
            visibility = bundle["visibility"].copy() if "visibility" in bundle else None
        rendered = render_tracks(_media_frames(source), tracks, visibility=visibility, trace_length=args.trace_length)
        import imageio.v2 as iio

        with iio.get_writer(output, fps=args.fps, codec="libx264") as writer:
            for frame in rendered:
                writer.append_data(frame)
        frame_count = len(rendered)

    payload = {
        "kind": args.kind,
        "artifact": str(artifact),
        "media": str(source) if source is not None else None,
        "output": str(output),
        "frame_count": frame_count,
    }
    if args.json:
        json_dump(payload)
    else:
        print(f"visualization: {output} ({args.kind}, {frame_count} frame{'s' if frame_count != 1 else ''})")
    return 0


def register_model_subparsers(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    from worldfoundry.base_models.capabilities import BASE_MODEL_CAPABILITIES, BASE_MODEL_STACKS

    models_parser = subparsers.add_parser("models", help="Inspect registered model runners")
    models_subparsers = models_parser.add_subparsers(dest="models_command", required=True)

    models_list_parser = models_subparsers.add_parser("list", help="List available model types")
    models_list_parser.add_argument("--family")
    models_list_parser.add_argument("--json", action="store_true")
    models_list_parser.set_defaults(func=_handle_models_list)

    models_runtime_runners_parser = models_subparsers.add_parser(
        "runtime-runners", help="List registered runner targets usable with evaluate --mode model"
    )
    models_runtime_runners_parser.add_argument("--json", action="store_true")
    models_runtime_runners_parser.set_defaults(func=_handle_models_runtime_runners)

    models_visualizations_parser = models_subparsers.add_parser(
        "visualizations", help="List official visualization support for in-tree base models"
    )
    models_visualizations_parser.add_argument("--family", choices=("three_dimensions", "perception_core"))
    models_visualizations_parser.add_argument("--model")
    models_visualizations_parser.add_argument("--json", action="store_true")
    models_visualizations_parser.set_defaults(func=_handle_models_visualizations)

    models_assets_parser = models_subparsers.add_parser(
        "assets",
        help="Plan or download reusable base-model assets",
        description="Plan or download reusable base-model assets such as depth, SLAM, detection, segmentation, and motion stacks.",
    )
    models_assets_parser.add_argument(
        "--capability",
        action="append",
        choices=sorted([*BASE_MODEL_CAPABILITIES, *BASE_MODEL_STACKS]),
        help="Capability or stack id to materialize. May repeat. Defaults to all registered capabilities.",
    )
    models_assets_parser.add_argument(
        "--list", action="store_true", help="List registered base-model capabilities and stacks."
    )
    models_assets_parser.add_argument("--execute-downloads", action="store_true")
    models_assets_parser.add_argument("--report-path", type=Path)
    models_assets_parser.add_argument("--json", action="store_true")
    models_assets_parser.set_defaults(func=_handle_models_assets)

    models_visualize_parser = models_subparsers.add_parser(
        "visualize",
        help="Render structured perception output using the shared official-style visualizers",
    )
    models_visualize_parser.add_argument(
        "--kind",
        required=True,
        choices=("detection", "mask", "flow", "tracks", "feature-pca", "depth", "normal", "keypoints", "text"),
    )
    models_visualize_parser.add_argument("--artifact", required=True, type=Path)
    models_visualize_parser.add_argument("--media", type=Path, help="Source image/video for overlays")
    models_visualize_parser.add_argument("--output", required=True, type=Path)
    models_visualize_parser.add_argument(
        "--normalized", action="store_true", help="Detection boxes use normalized xyxy"
    )
    models_visualize_parser.add_argument("--inverse", action="store_true", help="Visualize inverse depth/disparity")
    models_visualize_parser.add_argument("--alpha", type=float, default=0.5)
    models_visualize_parser.add_argument("--trace-length", type=int, default=12)
    models_visualize_parser.add_argument("--fps", type=float, default=10.0)
    models_visualize_parser.add_argument("--json", action="store_true")
    models_visualize_parser.set_defaults(func=_handle_models_visualize)
