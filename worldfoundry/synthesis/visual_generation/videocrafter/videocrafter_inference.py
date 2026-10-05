"""Inference helpers for the packaged VideoCrafter runtime."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from worldfoundry.base_models.diffusion_model.schedulers.lvdm.ddim import (
    DDIMSampler,
)


def batch_ddim_sampling(
    model,
    cond,
    noise_shape,
    n_samples: int = 1,
    ddim_steps: int = 50,
    ddim_eta: float = 1.0,
    cfg_scale: float = 1.0,
    temporal_cfg_scale=None,
    **kwargs,
):
    """Batch ddim sampling.

    Args:
        model: The model.
        cond: The cond.
        noise_shape: The noise shape.
        n_samples: The n samples.
        ddim_steps: The ddim steps.
        ddim_eta: The ddim eta.
        cfg_scale: The cfg scale.
        temporal_cfg_scale: The temporal cfg scale.
    """
    ddim_sampler = DDIMSampler(model)
    uncond_type = model.uncond_type
    batch_size = noise_shape[0]
    if "fs" not in kwargs and isinstance(cond, dict) and "fps" in cond:
        kwargs["fs"] = cond["fps"]

    if cfg_scale != 1.0:
        if uncond_type == "empty_seq":
            uc_emb = model.get_learned_conditioning(batch_size * [""])
        elif uncond_type == "zero_embed":
            c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
            uc_emb = torch.zeros_like(c_emb)
        else:
            raise ValueError(f"Unsupported unconditional embedding type: {uncond_type!r}")

        if hasattr(model, "embedder"):
            uc_img = torch.zeros(noise_shape[0], 3, 224, 224).to(model.device)
            uc_img = model.get_image_embeds(uc_img)
            uc_emb = torch.cat([uc_emb, uc_img], dim=1)

        if isinstance(cond, dict):
            uc = {key: cond[key] for key in cond.keys()}
            uc.update({"c_crossattn": [uc_emb]})
        else:
            uc = uc_emb
    else:
        uc = None

    batch_variants = []
    for _ in range(n_samples):
        kwargs.update({"clean_cond": True})
        samples, _ = ddim_sampler.sample(
            S=ddim_steps,
            conditioning=cond,
            batch_size=noise_shape[0],
            shape=noise_shape[1:],
            verbose=False,
            unconditional_guidance_scale=cfg_scale,
            unconditional_conditioning=uc,
            eta=ddim_eta,
            temporal_length=noise_shape[2],
            conditional_guidance_scale_temporal=temporal_cfg_scale,
            x_T=None,
            **kwargs,
        )
        batch_variants.append(model.decode_first_stage_2DAE(samples))
    return torch.stack(batch_variants, dim=1)


def _normalized_checkpoint_state_dict(payload):
    """Helper function to normalized checkpoint state dict.

    Args:
        payload: The payload.
    """
    if isinstance(payload, dict) and "module" in payload and hasattr(payload["module"], "keys"):
        new_pl_sd = OrderedDict()
        for key in payload["module"].keys():
            new_pl_sd[key[16:]] = payload["module"][key]
        return new_pl_sd
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    return payload


def load_model_checkpoint(model, ckpt: str):
    """Load model checkpoint.

    Args:
        model: The model.
        ckpt: The ckpt.
    """
    state_dict = _normalized_checkpoint_state_dict(
        torch.load(ckpt, map_location="cpu", weights_only=True)
    )
    expected = model.state_dict()
    for key in ("scale_arr_prev",):
        if key in expected and key not in state_dict:
            state_dict[key] = expected[key]
    model.load_state_dict(state_dict, strict=True)
    print(">>> model checkpoint loaded.")
    return model


def load_image_batch(filepath_list, image_size=(256, 256)):
    """Load image batch.

    Args:
        filepath_list: The filepath list.
        image_size: The image size.
    """
    batch_tensor = []
    for filepath in filepath_list:
        path = Path(filepath)
        suffix = path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            rgb_img = Image.open(path).convert("RGB")
            rgb_img = rgb_img.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)
            img_tensor = torch.from_numpy(np.asarray(rgb_img, np.float32)).permute(2, 0, 1).float()
        else:
            raise NotImplementedError(
                f"VideoCrafter image conditioning supports image files only, got {suffix!r}."
            )
        img_tensor = (img_tensor / 255.0 - 0.5) * 2
        batch_tensor.append(img_tensor)
    return torch.stack(batch_tensor, dim=0)
