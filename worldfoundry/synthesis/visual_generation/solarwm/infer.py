"""Launcher for SolarWM Wan2.2-5B inference through the vendored solarwm package."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

from worldfoundry.core.execution.process import package_module_command
from worldfoundry.synthesis.visual_generation.solarwm.runtime import SOLARWM_MODULE


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--test-index")
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _relative_test_index(value: str) -> str:
    text = str(value or "")
    path = PurePosixPath(text)
    if (
        not text
        or text.startswith("/")
        or "://" in text
        or "\\" in text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("SolarWM test-index must be a portable relative POSIX path")
    return path.as_posix()


def build_official_command(args: argparse.Namespace, *, run_root: Path) -> list[str]:
    """Return an argv-safe SolarWM command with explicit paths."""
    if args.sample_count != 1:
        raise ValueError("SolarWM's single-MP4 WorldFoundry binding requires sample-count=1")
    arguments = ["infer", "--config", str(args.config.expanduser().resolve())]
    overrides = [
        f"model.base_path={args.base_model_dir.expanduser().resolve()}",
        f"checkpoint.path={args.checkpoint_dir.expanduser().resolve()}",
        f"data.index_root={args.data_root.expanduser().resolve()}",
        f"data.transport.root={args.data_root.expanduser().resolve()}",
        f"runtime.output_dir={run_root}",
        "inference.output_layout=transaction_v1",
        f"validation.sample_count={args.sample_count}",
        f"validation.selection_seed={args.seed}",
        f"validation.noise_seed={args.seed}",
        f"data.seed={args.seed}",
    ]
    if args.test_index:
        overrides.append(f"data.test_index={_relative_test_index(args.test_index)}")
    for override in overrides:
        arguments.extend(["--set", override])
    return package_module_command(SOLARWM_MODULE, arguments)


def _find_generated_video(run_root: Path) -> Path:
    """Select the final official video artifact from a one-sample transaction."""
    candidates = sorted(run_root.glob("generation/*/slot-*/video.mp4"))
    if not candidates:
        candidates = sorted(run_root.glob("generation/**/video.mp4"))
    candidates = [path for path in candidates if path.is_file() and path.stat().st_size > 0]
    if not candidates:
        raise RuntimeError("SolarWM inference completed without producing generation/**/video.mp4")
    return candidates[-1]


def run(args: argparse.Namespace) -> Path:
    """Launch the SolarWM module and normalize its artifact into one MP4."""
    if args.sample_count != 1:
        raise ValueError("SolarWM's single-MP4 WorldFoundry binding requires sample-count=1")

    output_path = args.output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_path.stem}-solarwm-", dir=output_path.parent) as temp_dir:
        run_root = Path(temp_dir) / "run"
        command = build_official_command(args, run_root=run_root)
        completed = subprocess.run(
            command,
            cwd=temp_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.stdout:
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        if completed.returncode != 0:
            tail = (completed.stdout or "")[-4000:]
            raise RuntimeError(f"SolarWM inference exited with code {completed.returncode}:\n{tail}")
        generated = _find_generated_video(run_root)
        shutil.copy2(generated, output_path)
    print(f"SolarWM output: {output_path.name}")
    return output_path


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
