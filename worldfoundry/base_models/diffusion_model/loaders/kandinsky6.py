"""Construct Kandinsky-6 components through the native checkpoint resolver."""

from pathlib import Path
import torch
from ..recipes.kandinsky6_config import PipelineConfig, load_config, _CONFIGS
from ..schedulers.kandinsky6.sampler_config import dit_class
from ..optimizations.kandinsky6.kernels import bind_attention
from . import NativeCheckpointResolver
from .kandinsky6_weights import load_bf16_checkpoint, materialize_meta_buffers


def component_config(context):
    cfg = load_config(_CONFIGS / "checkpoints" / (str(context.recipe_options["checkpoint_name"]) + ".yaml"))
    strategy = context.policy.offload.mode.value
    strategy = {"component": "module", "cpu": "module"}.get(strategy, strategy)
    if strategy not in {"none", "module", "block"}:
        raise ValueError(f"Kandinsky-6 does not support offload mode {strategy!r}")
    cfg = cfg.model_copy(update={"offload": cfg.offload.model_copy(update={"strategy": strategy})})
    root = NativeCheckpointResolver().materialize(context.require_checkpoint("weights")).root
    cfg = cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(
                update={key: str(root / value) for key, value in cfg.paths.model_dump().items() if value is not None}
            )
        }
    )
    if cfg.audio_vae is not None:
        cfg = cfg.model_copy(
            update={
                "audio_vae": cfg.audio_vae.model_copy(update={"tod_vae_ckpt": str(root / cfg.audio_vae.tod_vae_ckpt)}),
                "vocoder": cfg.vocoder.model_copy(update={"ckpt": str(root / cfg.vocoder.ckpt)}),
            }
        )
    load_device = torch.device("cpu") if strategy != "none" else context.policy.device
    return cfg, load_device


def create_bare_dit(
    cfg: PipelineConfig,
    device: str | torch.device,
    attention_engine: str,
):
    """Construct DiT and load its checkpoint — no CacheDiT / compile / AOTI.

    Builds on the ``meta`` device so we never allocate the full float32 skeleton
    (~120 GB / ~2 min for Pro T2VA) before overwriting it from the checkpoint.
    """
    device = torch.device(device)
    dit_cfg = cfg.dit
    with torch.device("meta"):
        dit_type = dit_class(cfg.piflow.enabled)
        dit_kwargs = dict(
            in_visual_dim=dit_cfg.in_visual_dim,
            out_visual_dim=dit_cfg.out_visual_dim,
            in_text_dim=dit_cfg.in_text_dim,
            in_text_dim2=dit_cfg.in_text_dim2,
            time_dim=dit_cfg.time_dim,
            patch_size=dit_cfg.patch_size,
            model_dim=dit_cfg.model_dim,
            ff_dim=dit_cfg.ff_dim,
            num_text_blocks=dit_cfg.num_text_blocks,
            num_visual_blocks=dit_cfg.num_visual_blocks,
            axes_dims=dit_cfg.axes_dims,
            visual_cond=dit_cfg.visual_cond,
            is_multimodal=dit_cfg.is_multimodal,
            in_audio_dim=dit_cfg.in_audio_dim,
            model_dim_a=dit_cfg.model_dim_a,
            time_dim_a=dit_cfg.time_dim_a,
            ff_dim_a=dit_cfg.ff_dim_a,
            axes_dims_a=dit_cfg.axes_dims_a,
            audio_freqs_scaling=dit_cfg.audio_freqs_scaling,
            attention_engine=attention_engine,
            text_token_padding=dit_cfg.text_token_padding,
            ca_rope=dit_cfg.ca_rope,
            cross_gates=dit_cfg.cross_gates,
            fix_modulation=dit_cfg.fix_modulation,
            visual_token_type_num_embeddings=dit_cfg.visual_token_type_num_embeddings,
        )
        if cfg.piflow.enabled:
            dit_kwargs.update(
                n_grid=cfg.piflow.dx_num_grid_points,
                out_visual_dim=dit_cfg.out_visual_dim,
                out_audio_dim=dit_cfg.in_audio_dim,
            )
        dit = dit_type(**dit_kwargs)

    load_bf16_checkpoint(dit, cfg.paths.dit)

    materialize_meta_buffers(dit, torch.device("cpu"))
    dit = dit.to(device).eval()
    return dit
