# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-backed WonderWorld replay through the resolved public loader."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image


def main() -> None:
    """Generate real splats and assert artifact and novel-view contracts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-path", type=Path)
    parser.add_argument("--prompt", default="A detailed photographic view of the scene.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth-steps", type=int, default=30)
    parser.add_argument("--inpaint-steps", type=int, default=50)
    parser.add_argument("--normal-steps", type=int, default=10)
    parser.add_argument("--refinement-steps", type=int, default=100)
    parser.add_argument("--repeat", action="store_true", help="Repeat A after a changed camera request B")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import torch.version
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

    from worldfoundry.evaluation.api import GenerationRequest
    from worldfoundry.evaluation.models.catalog import load_model_zoo_registry
    from worldfoundry.evaluation.models.pipelines.invocation import build_pipeline_invocation, invoke_pipeline
    from worldfoundry.evaluation.models.pipelines.loading import PipelineRunnerSpec, load_pipeline_from_spec

    entry = load_model_zoo_registry().get("wonderworld")
    assert entry.pipeline_target is not None
    pipeline = load_pipeline_from_spec(
        PipelineRunnerSpec(
            model_id="wonderworld",
            pipeline_target=entry.pipeline_target,
            runtime_profile_id="wonderworld",
            model_path={"checkpoint_root": str(args.checkpoint_root)},
            fallback_model_path=None,
            required_components=None,
            device="cuda",
        )
    )
    args.output.mkdir(parents=True, exist_ok=True)
    cameras = (
        np.asarray(json.loads(args.camera_path.read_text()), dtype=np.float32)
        if args.camera_path
        else np.empty((0, 4, 4), dtype=np.float32)
    )
    requests = [("A", cameras)]
    if args.repeat:
        changed = np.eye(4, dtype=np.float32)[None]
        changed[0, 0, 3] = -0.6
        requests.extend((("B", changed), ("A_repeat", cameras)))
    reports: list[dict[str, Any]] = []
    for name, poses in requests:
        torch.cuda.synchronize()
        started = perf_counter()
        invocation = build_pipeline_invocation(
            request=GenerationRequest(
                sample_id=name,
                task_name="image-to-3d-world",
                inputs={"prompt": args.prompt, "image": str(args.image)},
                controls={"camera_path": poses.tolist()},
                generation_kwargs={
                    "seed": args.seed,
                    "depth_steps": args.depth_steps,
                    "inpaint_steps": args.inpaint_steps,
                    "normal_steps": args.normal_steps,
                    "refinement_steps": args.refinement_steps,
                    "run_dir": args.output / name,
                },
            ),
            output_dir=args.output / name,
            artifact_filename="world.splat",
        )
        with sdpa_kernel(SDPBackend.MATH):
            result = invoke_pipeline(pipeline, invocation)
        torch.cuda.synchronize()
        duration = perf_counter() - started
        artifact = Path(result["artifact_path"])
        records = np.fromfile(
            artifact,
            dtype=[("position", "<f4", (3,)), ("scale", "<f4", (3,)), ("rgba", "u1", (4,)), ("rotation", "u1", (4,))],
        )
        assert artifact.stat().st_size == 32 * result["metadata"]["surfel_count"] > 0
        assert np.isfinite(records["position"]).all() and np.isfinite(records["scale"]).all()
        assert (records["scale"][:, :2] > 0).all()
        assert (records["scale"][:, 2] >= 0).all()
        preview = np.asarray(Image.open(result["preview_path"]))
        assert preview.shape == (512, 512, 3) and preview.std() > 1
        assert len(result["metadata"]["views"]) == len(poses) + 1
        view_stats = []
        for view in result["metadata"]["views"]:
            index = view["index"]
            rendered = np.asarray(Image.open(Path(result["run_dir"]) / f"view_{index:03d}.png"))
            assert rendered.shape == (512, 512, 3) and rendered.std() > 1
            if view["new_pixels"]:
                depth = np.load(Path(result["run_dir"]) / f"depth_{index:03d}.npy")
                assert depth.shape == (512, 512) and np.isfinite(depth).all() and (depth > 0).all()
            view_stats.append(
                {"index": index, "preview_std": float(rendered.std()), "new_surfels": view["new_surfels"]}
            )
        report = {
            "request": name,
            "seconds": duration,
            "artifact_bytes": artifact.stat().st_size,
            "preview_std": float(preview.std()),
            "views": view_stats,
            "result": result,
        }
        reports.append(report)
        print(json.dumps(report), flush=True)
    if args.repeat:
        exports = [Path(report["result"]["artifact_path"]).read_bytes() for report in reports]
        assert exports[0] == exports[2]
        assert exports[0] != exports[1]
    contract = {
        "status": "passed",
        "scope": "native checkpoint-backed artifact replay; not official full-pipeline parity",
        "invocation": "standard-evaluation",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "requests": reports,
        "attention_backend": "math",
        "deterministic_algorithms": True,
        "deterministic_warn_only": True,
        "refinement_steps": args.refinement_steps,
        "repeat_equal": True if args.repeat else None,
    }
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")


if __name__ == "__main__":
    main()
