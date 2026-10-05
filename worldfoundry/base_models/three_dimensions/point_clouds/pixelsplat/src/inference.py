"""Pure PyTorch inference entry point for pixelSplat."""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from worldfoundry.core.io.paths import resolve_data_path

from .config import load_inference_config
from .dataset import get_dataset
from .global_cfg import set_cfg
from .misc.image_io import save_image
from .model.decoder import get_decoder
from .model.encoder import get_encoder
from .model.ply_export import export_ply


def _load_component_state(module: nn.Module, state: dict[str, torch.Tensor], prefix: str) -> None:
    component = {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not component:
        if not module.state_dict():
            return
        raise RuntimeError(f"Checkpoint contains no parameters with prefix {prefix!r}.")
    module.load_state_dict(component, strict=True)


@torch.inference_mode()
def run_inference(cfg_dict: DictConfig) -> None:
    """Load a checkpoint and render the configured posed-image examples."""
    set_cfg(cfg_dict)
    cfg = load_inference_config(cfg_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    _load_component_state(encoder, state, "encoder.")
    _load_component_state(decoder, state, "decoder.")
    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()

    data_shim = encoder.get_data_shim() if hasattr(encoder, "get_data_shim") else lambda batch: batch
    dataset = get_dataset(cfg.dataset, "test")
    loader = DataLoader(dataset, batch_size=1, num_workers=cfg.num_workers)

    for batch in loader:
        batch = data_shim(batch)
        context = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch["context"].items()
        }
        target = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch["target"].items()
        }
        gaussians = encoder(context, 0, deterministic=False)
        _, _, _, height, width = target["image"].shape
        colors = []
        for start in range(0, target["far"].shape[1], 32):
            output = decoder(
                gaussians,
                target["extrinsics"][:, start : start + 32],
                target["intrinsics"][:, start : start + 32],
                target["near"][:, start : start + 32],
                target["far"][:, start : start + 32],
                (height, width),
            )
            colors.append(output.color)
        color = torch.cat(colors, dim=1)

        scene = batch["scene"][0]
        scene_dir = cfg.output_path / scene
        for index, image in zip(target["index"][0], color[0]):
            save_image(image, scene_dir / "color" / f"{int(index):06d}.png")
        for index, image in zip(context["index"][0], context["image"][0]):
            save_image(image, scene_dir / "context" / f"{int(index):06d}.png")
        if gaussians.scales is not None and gaussians.rotations is not None:
            export_ply(
                context["extrinsics"][0, 0],
                gaussians.means[0],
                gaussians.scales[0],
                gaussians.rotations[0],
                gaussians.harmonics[0],
                gaussians.opacities[0],
                scene_dir / "gaussians.ply",
            )


@hydra.main(
    version_base=None,
    config_path=str(resolve_data_path("models", "runtime", "configs", "pixelsplat", "config")),
    config_name="main",
)
def main(cfg_dict: DictConfig) -> None:
    run_inference(cfg_dict)


if __name__ == "__main__":
    main()
