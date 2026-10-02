"""Assemble GEN3C and Lyra-1 checkpoints for the public pipeline."""

import shutil
from pathlib import Path
from typing import Optional

from worldfoundry.base_models.three_dimensions.point_clouds.lyra.utils import (
    _candidate_lyra1_weights_subdir,
    default_local_lyra1_checkpoint_root,
)
from worldfoundry.core.io.paths import scratch_directory


def _lyra1_checkpoint_layout_complete(root: Path) -> bool:
    """Lyra1 checkpoint layout complete helper function."""
    required_files = [
        root / "Gen3C-Cosmos-7B" / "model.pt",
        root / "Cosmos-Tokenize1-CV8x8x8-720p" / "mean_std.pt",
        root / "google-t5" / "t5-11b" / "config.json",
        root / "Lyra" / "lyra_static.pt",
        root / "Lyra" / "lyra_dynamic.pt",
    ]
    return all(path.exists() for path in required_files)


def _reset_dir(path: Path):
    """Reset dir helper function."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        # Safely clean up any pre-existing output directory structure
        shutil.rmtree(path)


def _link_or_copy(src: Path, dst: Path):
    """Link or copy helper function."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        _reset_dir(dst)
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            # Copy source file to target directory while preserving original metadata
            shutil.copy2(src, dst)


def prepare_lyra1_checkpoint_root(
    checkpoint_dir: Optional[str],
    repo_root: Optional[str] = None,
) -> str:
    """Prepare lyra1 checkpoint root helper function."""
    from worldfoundry.synthesis.visual_generation.gen3c.runtime_env import (
        prepare_gen3c_checkpoint_root,
    )

    candidates = []
    if checkpoint_dir is not None:
        candidates.append(Path(checkpoint_dir).expanduser())
    if repo_root is not None:
        candidates.append(Path(repo_root).expanduser() / "checkpoints")

    for candidate in candidates:
        if candidate.exists() and _lyra1_checkpoint_layout_complete(candidate):
            return str(candidate.resolve())

    lyra_weights_root = None
    for candidate in candidates:
        if candidate.exists():
            lyra_weights_root = _candidate_lyra1_weights_subdir(candidate)
            if lyra_weights_root is not None:
                break
    if lyra_weights_root is None:
        lyra_weights_root = default_local_lyra1_checkpoint_root()
    if lyra_weights_root is None:
        raise FileNotFoundError(
            "Unable to locate local Lyra-1 weights. "
            "Expected a directory containing 'Lyra/lyra_static.pt' and 'Lyra/lyra_dynamic.pt', "
            "or pass required_components['checkpoint_dir'] pointing to that layout."
        )

    gen3c_checkpoint_root = (
        Path(
            prepare_gen3c_checkpoint_root(
                checkpoint_dir=checkpoint_dir,
            )
        )
        .expanduser()
        .resolve()
    )

    if _lyra1_checkpoint_layout_complete(gen3c_checkpoint_root):
        return str(gen3c_checkpoint_root)

    stage_root = scratch_directory("lyra1_checkpoints_").resolve()

    for item in gen3c_checkpoint_root.iterdir():
        if item.name == "Lyra":
            continue
        _link_or_copy(item, stage_root / item.name)

    target_lyra_dir = stage_root / "Lyra"
    if lyra_weights_root.name == "Lyra":
        _link_or_copy(lyra_weights_root, target_lyra_dir)
    else:
        target_lyra_dir.mkdir(parents=True, exist_ok=True)
        _link_or_copy(lyra_weights_root / "lyra_static.pt", target_lyra_dir / "lyra_static.pt")
        _link_or_copy(lyra_weights_root / "lyra_dynamic.pt", target_lyra_dir / "lyra_dynamic.pt")

    return str(stage_root.resolve())
