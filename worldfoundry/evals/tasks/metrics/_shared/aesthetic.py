"""Shared LAION aesthetic-predictor head loaders.

Single in-tree source for the two aesthetic heads that benchmark runtimes
(VBench, MiraBench, WBench, WorldArena, MemoBench, ...) previously each
vendored their own copy of:

- linear head — ``sa_0_4_vit_l_14_linear.pth`` → ``nn.Linear(768, 1)`` scoring
  L2-normalized CLIP ViT-L/14 image features.
- MLP head — ``sac+logos+ava1-l14-linearMSE.pth`` (improved-aesthetic-predictor)
  scoring the same features through a small MLP.

Checkpoint resolution fails loudly with the searched paths; there is no silent
proxy score when a checkpoint is missing.
"""

from __future__ import annotations

from pathlib import Path

LINEAR_HEAD_FILENAME = "sa_0_4_vit_l_14_linear.pth"

#: Relative locations tried when a directory is passed instead of a file.
_LINEAR_HEAD_DIR_CANDIDATES = (
    LINEAR_HEAD_FILENAME,
    f"aesthetic_model/{LINEAR_HEAD_FILENAME}",
    f"aesthetic_model/emb_reader/{LINEAR_HEAD_FILENAME}",
)


def resolve_laion_aesthetic_linear_checkpoint(checkpoint: str | Path | None = None) -> Path:
    """Resolve the LAION linear-head checkpoint path or fail loudly.

    ``checkpoint`` may be the ``.pth`` file itself or a staging directory that
    contains it under one of the conventional layouts. When omitted, the shared
    VBench asset registration is used.
    """
    if checkpoint is None:
        from worldfoundry.base_models.capabilities import vbench_asset_path

        return Path(vbench_asset_path("vbench_aesthetic_linear_checkpoint"))
    path = Path(checkpoint).expanduser()
    if path.is_file():
        return path
    candidates = [path / relative for relative in _LINEAR_HEAD_DIR_CANDIDATES]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = "\n".join(f"  - {candidate}" for candidate in (path, *candidates))
    raise FileNotFoundError(
        f"LAION aesthetic linear head ({LINEAR_HEAD_FILENAME}) is not staged.\nSearched:\n{searched}"
    )


def load_laion_aesthetic_linear_head(checkpoint: str | Path | None = None):
    """Load the inference-only LAION aesthetic linear head (768 → 1)."""
    import torch
    from torch import nn

    path = resolve_laion_aesthetic_linear_checkpoint(checkpoint)
    model = nn.Linear(768, 1)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model.eval()


def load_laion_aesthetic_mlp_head(checkpoint: str | Path):
    """Load the improved-aesthetic-predictor MLP head (768 → 1).

    ``checkpoint`` must point at the staged ``sac+logos+ava1-l14-linearMSE.pth``
    weights; callers own any download/staging protocol.
    """
    import torch
    from torch import nn

    path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"LAION aesthetic MLP head is not staged: {path}")
    model = nn.Sequential(
        nn.Linear(768, 1024),
        nn.Dropout(0.2),
        nn.Linear(1024, 128),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.Dropout(0.1),
        nn.Linear(64, 16),
        nn.Linear(16, 1),
    )
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    # Upstream checkpoint stores keys as "layers.<i>.*"; nn.Sequential expects "<i>.*".
    state_dict = {key.replace("layers.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    return model.eval()


__all__ = [
    "LINEAR_HEAD_FILENAME",
    "load_laion_aesthetic_linear_head",
    "load_laion_aesthetic_mlp_head",
    "resolve_laion_aesthetic_linear_checkpoint",
]
