"""Run the complete MoVerse inference pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from worldfoundry.core.execution.process import package_module_command

RUNTIME_ROOT = Path(__file__).resolve().parents[1]


def _run(python: str, module: str, *arguments: object) -> None:
    subprocess.run(
        package_module_command(module, [str(argument) for argument in arguments], python_executable=python),
        cwd=RUNTIME_ROOT,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoVerse image-to-world inference")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--back_prompt", default="")
    parser.add_argument("--traj", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gimbal360_dir", required=True)
    parser.add_argument("--flux_model_path", required=True)
    parser.add_argument("--da3_model_dir", required=True)
    parser.add_argument("--gaussian_checkpoint", required=True)
    parser.add_argument("--video_checkpoint", required=True)
    parser.add_argument("--taehv_checkpoint", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--default_config_path", required=True)
    parser.add_argument("--num_frames", type=int, default=161)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--python_executable", default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Path(args.image).expanduser().resolve()
    stem = image.stem

    trajectory = Path(args.traj).expanduser().resolve() if args.traj else output_dir / "trajectory.txt"
    if not args.traj:
        trajectory.write_text("0 0\n0 0\n0 0\n", encoding="utf-8")

    panorama_args: list[object] = [
        "--image",
        image,
        "--prompt",
        args.prompt,
        "--ckpt_dir",
        args.gimbal360_dir,
        "--basemodel_name_or_path",
        args.flux_model_path,
        "--output_dir",
        output_dir,
    ]
    if args.back_prompt:
        panorama_args.extend(["--back_prompt", args.back_prompt])
    seed_args: list[object] = [] if args.seed is None else ["--seed", args.seed]
    _run(
        args.python_executable,
        "worldfoundry.synthesis.visual_generation.moverse.moverse_runtime.pano_gen.inference",
        *panorama_args,
        *seed_args,
    )

    panorama = output_dir / f"{stem}_pano.png"
    depth = output_dir / "depth.npy"
    scene = output_dir / f"{stem}_pano.ply"
    video = output_dir / "video.mp4"

    _run(
        args.python_executable,
        "worldfoundry.synthesis.visual_generation.moverse.moverse_runtime.gaussian_gen.infer_da3",
        "--path",
        panorama,
        "--output_dir",
        output_dir,
        "--model_dir",
        args.da3_model_dir,
    )
    _run(
        args.python_executable,
        "worldfoundry.synthesis.visual_generation.moverse.moverse_runtime.gaussian_gen.infer_pano",
        "--image",
        panorama,
        "--depth",
        depth,
        "--checkpoint",
        args.gaussian_checkpoint,
        "--output_dir",
        output_dir,
    )
    _run(
        args.python_executable,
        "worldfoundry.synthesis.visual_generation.moverse.moverse_runtime.scripts.run_pipeline",
        "--scene",
        scene,
        "--traj",
        trajectory,
        "--config_path",
        args.config_path,
        "--default_config_path",
        args.default_config_path,
        "--checkpoint_path",
        args.video_checkpoint,
        "--taehv",
        "--taehv_checkpoint",
        args.taehv_checkpoint,
        "--prompt",
        f"A video of {args.prompt}",
        "--output",
        video,
        "--num_frames",
        args.num_frames,
        "--fps",
        args.fps,
        *seed_args,
    )


if __name__ == "__main__":
    main()
