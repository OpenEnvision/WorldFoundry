from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldFoundry FramePack non-interactive official runner")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--hf-home", required=True)
    parser.add_argument("--hunyuan-root", default=None)
    parser.add_argument("--flux-redux-root", default=None)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--latent-window-size", type=int, default=9)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=31337)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--gs", type=float, default=10.0)
    parser.add_argument("--rs", type=float, default=0.0)
    parser.add_argument("--gpu-memory-preservation", type=float, default=6.0)
    parser.add_argument("--use-teacache", nargs="?", const=True, type=_parse_bool, default=True)
    parser.add_argument("--no-use-teacache", dest="use_teacache", action="store_false")
    parser.add_argument("--mp4-crf", type=int, default=16)
    return parser.parse_args()


def _resolve_local_model(value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.exists():
        return None
    if (path / "snapshots").is_dir():
        refs_main = path / "refs" / "main"
        if refs_main.is_file():
            revision = refs_main.read_text(encoding="utf-8").strip()
            snapshot = path / "snapshots" / revision
            if snapshot.exists():
                return str(snapshot.resolve())
        snapshots = sorted((path / "snapshots").iterdir())
        for snapshot in reversed(snapshots):
            if snapshot.is_dir():
                return str(snapshot.resolve())
    return str(path.resolve())


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    # The published F1 checkpoint uses forward section ordering and memory
    # layout, whereas FramePackI2V_HY uses the original backward schedule.
    os.environ["WORLDFOUNDRY_FRAMEPACK_VARIANT"] = (
        "f1" if any(re.search(r"framepack[-_]f1", part.lower()) for part in checkpoint_path.parts) else "original"
    )
    image_path = Path(args.image_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    hf_home = Path(args.hf_home).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"FramePack checkpoint path not found: {checkpoint_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"FramePack input image not found: {image_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    hunyuan_root = _resolve_local_model(args.hunyuan_root)
    flux_redux_root = _resolve_local_model(args.flux_redux_root)
    from .framepack_runtime import inference

    inference.initialize(
        checkpoint_path=checkpoint_path,
        output_dir=output_path.parent,
        hunyuan_root=hunyuan_root,
        flux_redux_root=flux_redux_root,
    )
    image = np.asarray(Image.open(image_path).convert("RGB"))
    produced = None
    for item in inference.process(
        image,
        args.prompt,
        "",
        int(args.seed),
        float(args.seconds),
        int(args.latent_window_size),
        int(args.steps),
        float(args.cfg),
        float(args.gs),
        float(args.rs),
        float(args.gpu_memory_preservation),
        bool(args.use_teacache),
        int(args.mp4_crf),
    ):
        if isinstance(item, str) and Path(item).is_file():
            produced = Path(item)
        elif isinstance(item, tuple) and item and isinstance(item[0], str) and Path(item[0]).is_file():
            produced = Path(item[0])
    if produced is None or not produced.is_file():
        raise RuntimeError("FramePack official process finished without producing an mp4.")
    shutil.copy2(produced, output_path)


if __name__ == "__main__":
    main()
