from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from worldfoundry.core.execution.process import package_module_command
from worldfoundry.core.io.paths import resolve_data_path

VARIANT_CONFIGS = {
    "4.5b-base": ("4.5B/4.5B_base_config.json", "ckpt/magi/4.5B_base"),
    "4.5b-distill": ("4.5B/4.5B_distill_config.json", "ckpt/magi/4.5B_distill"),
    "4.5b-distill-quant": ("4.5B/4.5B_distill_quant_config.json", "ckpt/magi/4.5B_distill_quant"),
}


def _magi_example_root() -> Path:
    override = os.getenv("WORLDFOUNDRY_MAGI_EXAMPLE_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return resolve_data_path("test_cases", "magi", "example").resolve()


def _special_token_path() -> Path:
    override = os.getenv("SPECIAL_TOKEN_PATH", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return _magi_example_root() / "assets" / "special_tokens.npz"


def _ffmpeg_bin_dir(shim_root: Path) -> str:
    """Return a bundled ffmpeg directory when system ffmpeg is unavailable."""

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return str(Path(system_ffmpeg).resolve().parent)
    try:
        import imageio_ffmpeg
    except Exception:
        return ""
    ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe()).expanduser()
    if not ffmpeg_exe.is_file():
        return ""
    shim_dir = shim_root / ".worldfoundry_bin"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim_path = shim_dir / "ffmpeg"
    if not shim_path.exists():
        try:
            shim_path.symlink_to(ffmpeg_exe)
        except OSError:
            shutil.copy2(ffmpeg_exe, shim_path)
            shim_path.chmod(0o755)
    return str(shim_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldFoundry MAGI-1 official runner")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--image-path", default="")
    parser.add_argument("--prefix-video-path", default="")
    parser.add_argument("--variant", default="4.5b-distill")
    parser.add_argument("--mode", choices=("t2v", "i2v", "v2v"), default="")
    parser.add_argument("--num-frames", type=int, default=96)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def _write_config(checkpoint_root: Path, output_dir: Path, args: argparse.Namespace) -> Path:
    key = str(args.variant).lower()
    if key not in VARIANT_CONFIGS:
        raise ValueError(f"Unsupported MAGI variant {args.variant!r}; choose one of {sorted(VARIANT_CONFIGS)}")
    config_rel, weights_rel = VARIANT_CONFIGS[key]
    config_path = _magi_example_root() / config_rel
    weights_path = checkpoint_root / weights_rel
    if not config_path.is_file():
        raise FileNotFoundError(f"MAGI official config not found: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"MAGI weights not found: {weights_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runtime = config["runtime_config"]
    runtime["load"] = str(weights_path)
    runtime["t5_pretrained"] = str(checkpoint_root / "ckpt" / "t5")
    runtime["vae_pretrained"] = str(checkpoint_root / "ckpt" / "vae")
    runtime["num_frames"] = int(args.num_frames)
    runtime["video_size_h"] = int(args.height)
    runtime["video_size_w"] = int(args.width)
    runtime["num_steps"] = int(args.num_steps)
    runtime["fps"] = int(args.fps)
    runtime["seed"] = int(args.seed)
    generated = output_dir / "magi_worldfoundry_config.json"
    generated.write_text(json.dumps(config, indent=4), encoding="utf-8")
    return generated


def main() -> None:
    args = parse_args()
    checkpoint_root = Path(args.checkpoint_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config_path = _write_config(checkpoint_root, output_path.parent, args)
    mode = args.mode or ("v2v" if args.prefix_video_path else "i2v" if args.image_path else "t2v")

    arguments = [
        "--config_file",
        str(config_path),
        "--mode",
        mode,
        "--prompt",
        args.prompt,
        "--output_path",
        str(output_path),
    ]
    if mode == "i2v":
        if not args.image_path:
            raise ValueError("MAGI i2v mode requires --image-path")
        arguments.extend(["--image_path", str(Path(args.image_path).expanduser().resolve())])
    if mode == "v2v":
        if not args.prefix_video_path:
            raise ValueError("MAGI v2v mode requires --prefix-video-path")
        arguments.extend(["--prefix_video_path", str(Path(args.prefix_video_path).expanduser().resolve())])

    env = os.environ.copy()
    env.setdefault("MASTER_ADDR", "localhost")
    env.setdefault("MASTER_PORT", "6009")
    env.setdefault("GPUS_PER_NODE", "1")
    env.setdefault("NNODES", "1")
    env.setdefault("WORLD_SIZE", "1")
    env.setdefault("PAD_HQ", "1")
    env.setdefault("PAD_DURATION", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("OFFLOAD_T5_CACHE", "true")
    env.setdefault("OFFLOAD_VAE_CACHE", "true")
    env["SPECIAL_TOKEN_PATH"] = str(_special_token_path())
    ffmpeg_dir = _ffmpeg_bin_dir(output_path.parent)
    if ffmpeg_dir:
        env["PATH"] = os.pathsep.join(item for item in (ffmpeg_dir, env.get("PATH", "")) if item)

    command = package_module_command(
        "worldfoundry.synthesis.visual_generation.magi.magi_runtime.inference.pipeline.entry", arguments
    )
    completed = subprocess.run(
        command,
        cwd=str(output_path.parent),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    log_path = output_path.with_suffix(output_path.suffix + ".magi.log")
    log_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"MAGI official runner failed with code {completed.returncode}; see {log_path}")
    if not output_path.is_file():
        candidates = sorted(output_path.parent.glob("*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not candidates:
            raise RuntimeError(f"MAGI official runner completed but no mp4 was found; see {log_path}")
        shutil.copy2(candidates[0], output_path)


if __name__ == "__main__":
    main()
