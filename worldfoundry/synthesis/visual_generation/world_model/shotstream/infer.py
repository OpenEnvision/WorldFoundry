"""Compatibility launcher for the official ShotStream inference script."""

from __future__ import annotations

import argparse
import csv
import json
import os
import runpy
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--wan-model-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--prompt", default="A cinematic scene with smooth natural motion.")
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _install_dependency_compatibility() -> None:
    """Apply narrow compatibility shims without editing the upstream checkout."""
    try:
        import pyarrow as pa

        if not hasattr(pa, "PyExtensionType") and hasattr(pa, "ExtensionType"):
            pa.PyExtensionType = pa.ExtensionType
    except ImportError:
        pass

    # Some vLLM environments expose an FA3 preview whose varlen function has a
    # newer required-argument signature than ShotStream's upstream call. The
    # released FA2 wheel implements the exact API used by ShotStream and runs on
    # Hopper, so make the upstream feature probe select that path.
    sys.modules["flash_attn_interface"] = None

    import torch
    import torchvision.io

    original_write_video = torchvision.io.write_video

    def compatible_write_video(filename: str, video_array: Any, fps: float, *args: Any, **kwargs: Any) -> None:
        try:
            original_write_video(filename, video_array, fps, *args, **kwargs)
            return
        except TypeError as exc:
            if "integer is required" not in str(exc):
                raise

        from worldfoundry.core.media.codecs.video import save_video_h264

        options = kwargs.get("options")
        try:
            crf = int(options.get("crf", 18)) if isinstance(options, dict) else 18
        except (TypeError, ValueError):
            crf = 18
        frames = torch.as_tensor(video_array, dtype=torch.uint8).detach().cpu().numpy()
        save_video_h264(frames, filename, fps=fps, crf=crf)

    torchvision.io.write_video = compatible_write_video


def _rewrite_input_csv(input_csv: Path, destination: Path) -> Path:
    """Copy an official CSV while resolving its relative asset paths."""
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    if not fieldnames:
        raise ValueError(f"ShotStream input CSV has no header: {input_csv}")
    for row in rows:
        for key in ("json_path", "video_path"):
            value = str(row.get(key) or "").strip()
            if value and not Path(value).expanduser().is_absolute():
                row[key] = str((input_csv.parent / value).resolve())
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def _write_prompt_fixture(prompt: str, workspace: Path) -> Path:
    fixture_dir = workspace / "input"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = fixture_dir / "prompt.json"
    prompt_path.write_text(
        json.dumps(
            {
                "global_caption": prompt.strip(),
                "shot1": "A smooth medium-wide tracking shot with coherent subject motion and stable scene details.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    csv_path = fixture_dir / "input.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("shot_num_from_caption", "json_path", "frame_number"))
        writer.writerow((1, str(prompt_path), "[[0, 81]]"))
    return csv_path


def _symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def run(args: argparse.Namespace) -> Path:
    source_dir = args.source_dir.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    wan_model_dir = args.wan_model_dir.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    upstream_entrypoint = source_dir / "Inference_Causal.py"
    if not upstream_entrypoint.is_file():
        raise FileNotFoundError(f"official ShotStream entrypoint is missing: {upstream_entrypoint}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_path.stem}-runtime-", dir=str(output_path.parent)) as temp_dir:
        workspace = Path(temp_dir)
        _symlink(checkpoint_dir, workspace / "ckpts")
        _symlink(wan_model_dir, workspace / "wan_models" / "Wan2.1-T2V-1.3B")
        generated_dir = workspace / "generated"
        generated_dir.mkdir()
        if args.input_csv is None:
            input_csv = _write_prompt_fixture(args.prompt, workspace)
        else:
            input_csv = _rewrite_input_csv(args.input_csv.expanduser().resolve(), workspace / "input.csv")

        _install_dependency_compatibility()
        previous_cwd = Path.cwd()
        previous_argv = list(sys.argv)
        source_text = str(source_dir)
        inserted_source = source_text not in sys.path
        if inserted_source:
            sys.path.insert(0, source_text)
        try:
            os.chdir(workspace)
            sys.argv = [
                str(upstream_entrypoint),
                "--config_path",
                str(checkpoint_dir / "shotstream.yaml"),
                "--output_folder",
                str(generated_dir),
                "--resume_ckpt",
                str(checkpoint_dir / "shotstream_merged.pt"),
                "--multi_caption",
                "True",
                "--data_path",
                str(input_csv),
                "--seed",
                str(args.seed),
            ]
            runpy.run_path(str(upstream_entrypoint), run_name="__main__")
        finally:
            os.chdir(previous_cwd)
            sys.argv = previous_argv
            if inserted_source:
                sys.path.remove(source_text)

        candidates = sorted(generated_dir.glob("*.mp4"), key=lambda path: path.stat().st_mtime)
        if not candidates:
            raise RuntimeError("official ShotStream inference completed without producing an MP4")
        shutil.copy2(candidates[-1], output_path)
    print(f"ShotStream output: {output_path.name}")
    return output_path


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
