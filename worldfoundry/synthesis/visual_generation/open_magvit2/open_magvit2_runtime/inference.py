"""Open-MAGVIT2 class-conditional image inference helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def load_model(config_path: str | Path, checkpoint_path: str | Path, device: str = "cuda"):
    """
    Load an Open-MAGVIT2 autoregressive sampler.

    Args:
        config_path: Path to an Open-MAGVIT2 YAML config.
        checkpoint_path: Path to the transformer checkpoint.
        device: Preferred torch device string.
    """
    import torch
    from omegaconf import OmegaConf

    from . import src as _src  # noqa: F401
    from worldfoundry.core.model_loading.factory import instantiate_from_config

    config = OmegaConf.load(str(config_path))
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    model = instantiate_from_config(config.model)
    model.load_state_dict(payload.get("state_dict"), strict=False)
    resolved_device = device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    model = model.to(resolved_device)
    model.eval()
    return model, config, resolved_device


def save_class_image(
    model: Any,
    config: Mapping[str, Any],
    output_path: str | Path,
    *,
    class_id: int,
    batch_size: int = 1,
    steps: int | None = None,
    temperature: tuple[float, float] = (1.0, 1.0),
    top_k: tuple[int, int] = (0, 0),
    top_p: tuple[float, float] = (0.96, 0.96),
    cfg_scale: tuple[float, float] = (4.0, 4.0),
) -> Path:
    """
    Generate one ImageNet class-conditional sample image.

    Args:
        model: Loaded Open-MAGVIT2 ``Net2NetTransformer`` instance.
        config: Parsed Open-MAGVIT2 config mapping.
        output_path: Target PNG path.
        class_id: ImageNet class label used by the class-conditional sampler.
        batch_size: Number of images to sample; the first one is written to ``output_path``.
        steps: Autoregressive token steps; defaults to config block size.
        temperature: Per-factor sampling temperature.
        top_k: Per-factor top-k values.
        top_p: Per-factor top-p values.
        cfg_scale: Per-factor classifier-free guidance scales.
    """
    import numpy as np
    import torch
    from einops import repeat
    from PIL import Image

    from src.Open_MAGVIT2.modules.transformer.gpt import sample_Open_MAGVIT2

    steps = int(steps or config.model.init_args.transformer_config.params.block_size)
    dim_z = int(config.model.init_args.first_stage_config.params.embed_dim)
    spatial_size = int(steps**0.5)
    if spatial_size * spatial_size != steps:
        raise ValueError(f"Open-MAGVIT2 expects square token grids, got {steps} steps.")

    labels = repeat(torch.tensor([class_id]), "1 -> b 1", b=batch_size).to(model.device)
    if cfg_scale[0] > 1.0:
        null_labels = torch.ones_like(labels) * model.transformer.config.class_num
        condition = torch.concat([labels, null_labels], dim=0)
    else:
        condition = labels

    with torch.no_grad():
        indices = sample_Open_MAGVIT2(
            condition,
            model.transformer,
            steps=steps,
            sample_logits=True,
            top_k=list(top_k),
            temperature=list(temperature),
            top_p=list(top_p),
            token_factorization=True,
            cfg_scale=list(cfg_scale),
        )
        samples = model.decode_to_img(indices, [batch_size, dim_z, spatial_size, spatial_size])

    image = samples[0].detach().cpu().numpy().transpose(1, 2, 0)
    image = (255 * ((image + 1.0) / 2.0)).clip(0, 255).astype(np.uint8)
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(target)
    return target
