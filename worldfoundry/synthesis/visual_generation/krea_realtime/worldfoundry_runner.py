from __future__ import annotations

import argparse
import os
import shutil
import sys
import types
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from worldfoundry.core.io.paths import package_data_path

DEFAULT_RUNTIME_ROOT = Path(__file__).resolve().parent / "krea_runtime"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldFoundry Krea realtime-video official runner")
    parser.add_argument("--repo-root", default=str(DEFAULT_RUNTIME_ROOT))
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--model-folder", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _write_config(repo_root: Path, checkpoint_path: Path, output_path: Path) -> Path:
    base_config = repo_root / "configs" / "self_forcing_server_14b.yaml"
    if not base_config.is_file():
        base_config = package_data_path('models', 'runtime', 'configs', 'krea_realtime', 'self_forcing_server_14b.yaml')
    if not base_config.is_file():
        raise FileNotFoundError(f"Krea official config not found: {base_config}")
    config = OmegaConf.load(base_config)
    config.checkpoint_path = str(checkpoint_path)
    generated = output_path.parent / "krea_realtime_worldfoundry.yaml"
    OmegaConf.save(config, generated)
    return generated


def _save_video_cv2(pixels, output_path: Path, fps: int = 16):
    import cv2

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = pixels[0].detach().cpu().permute(0, 2, 3, 1).clamp(0, 1).numpy()
    frames = (frames * 255).astype(np.uint8)
    height, width = frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open video writer for {output_path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return output_path


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Krea checkpoint not found: {checkpoint_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["MODEL_FOLDER"] = str(Path(args.model_folder).expanduser().resolve())
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if "dotenv" not in sys.modules:
        try:
            import dotenv  # noqa: F401
        except ModuleNotFoundError:
            shim = types.ModuleType("dotenv")
            shim.load_dotenv = lambda *_, **__: False
            sys.modules["dotenv"] = shim

    old_cwd = Path.cwd()
    os.chdir(repo_root)
    try:
        import sample
        from release_server import GenerateParams

        sample.save_video_direct = _save_video_cv2
        sample.save_video_ffmpeg_pipe = _save_video_cv2

        config_path = _write_config(repo_root, checkpoint_path, output_path)
        params = GenerateParams(
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            seed=args.seed,
            num_blocks=args.num_blocks,
        )
        results = sample.sample_videos(
            [args.prompt],
            config_path=str(config_path),
            output_dir=str(output_path.parent),
            params=params,
            save_videos=True,
            fps=args.fps,
        )
    finally:
        os.chdir(old_cwd)

    produced = results.get(0, {}).get("video_path") if isinstance(results, dict) else None
    produced_path = Path(produced).expanduser() if produced else None
    if produced_path is None or not produced_path.is_file():
        candidates = sorted(output_path.parent.glob("*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
        produced_path = candidates[0] if candidates else None
    if produced_path is None or not produced_path.is_file():
        raise RuntimeError("Krea official runner finished but no mp4 artifact was produced.")
    if produced_path.resolve() != output_path:
        shutil.copy2(produced_path, output_path)


if __name__ == "__main__":
    main()
