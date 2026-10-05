"""Resolve metric weights outside the source tree."""
import os
from pathlib import Path

from worldfoundry.core.io.paths import checkpoint_root_path, package_root


def get_weights_dir(subdir: str = "") -> str:
    """Get weight subdirectory path (auto-creates)."""
    root = Path(
        os.environ.get("HARNESSEVAL_WEIGHTS_ROOT", checkpoint_root_path("harnesseval-w"))
    )
    path = (root / subdir if subdir else root).expanduser().resolve()
    if path.is_relative_to(package_root()):
        raise ValueError("HarnessEval weights/cache must live outside the package source tree")
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def setup_torch_hub_dir():
    """Set torch.hub cache to project-local weights/torch_hub/."""
    import torch
    hub_dir = get_weights_dir("torch_hub")
    torch.hub.set_dir(hub_dir)
    return hub_dir


_WEIGHT_ASSETS = {
    ("amt", "amt-s.pth"): ("vbench_metric_checkpoint_assets", "vbench_amt_s_checkpoint"),
    ("raft", "raft-things.pth"): ("raft", "raft_things_checkpoint"),
    ("aesthetic", "sa_0_4_vit_l_14_linear.pth"): ("wbench_quality_metric_assets", "wbench_aesthetic_linear_checkpoint"),
    ("clip", "ViT-B-32.pt"): ("wbench_quality_metric_assets", "wbench_clip_vit_b32_checkpoint"),
    ("clip", "ViT-L-14.pt"): ("wbench_quality_metric_assets", "wbench_clip_vit_l14_checkpoint"),
}


def get_weight_file(subdir: str, filename: str, root: Path | None = None) -> str:
    """Use explicitly staged weights, then the existing base-model asset registry."""
    path = (root if root is not None else Path(get_weights_dir(subdir))) / filename
    if path.is_file():
        return str(path)
    from worldfoundry.base_models.capabilities import BASE_MODEL_CAPABILITIES

    capability, asset_id = _WEIGHT_ASSETS[(subdir, filename)]
    asset = next(asset for asset in BASE_MODEL_CAPABILITIES[capability].assets if asset.id == asset_id)
    matched = asset.check().get("matched_path")
    if matched:
        return str(matched)
    raise FileNotFoundError(f"Stage {path} or configure the registered base-model asset {asset_id}")
