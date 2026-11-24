"""Checkpoint loader for the in-tree MME-VLA inference policy."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from ..openpi import checkpoints, transforms
from ..openpi.modeling import model as _model
from . import config as _config
from .policy import MMEVLAPolicy


def create_policy(
    runtime_config: _config.RuntimeConfig,
    checkpoint_dir: Path | str,
    *,
    seed: int = 42,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
) -> MMEVLAPolicy:
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    params_dir = checkpoint_dir / "params"
    if not params_dir.is_dir():
        raise FileNotFoundError(f"MME-VLA params directory does not exist: {params_dir}")

    history_path = checkpoint_dir.parent / "history_config.txt"
    history_config = history_path.read_text(encoding="utf-8").strip() if history_path.is_file() else None
    if history_config in {"", "none", "None"}:
        history_config = None
    if runtime_config.model.history_config != history_config:
        runtime_config = dataclasses.replace(
            runtime_config,
            model=dataclasses.replace(
                runtime_config.model,
                history_config=history_config,
                use_history=history_config is not None,
            ),
        )

    logging.info("Loading MME-VLA parameters from %s", params_dir)
    params = _model.restore_params(params_dir, dtype=jnp.bfloat16)
    model = runtime_config.model.load(params)
    data_config = runtime_config.data.create(runtime_config.assets_dirs, runtime_config.model)

    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("MME-VLA config requires an asset id for normalization")
        norm_stats = checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    repack_transforms = repack_transforms or transforms.Group()
    return MMEVLAPolicy(
        model,
        seed=seed,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=runtime_config.policy_metadata,
        norm_stats=norm_stats,
        use_quantiles=data_config.use_quantile_norm,
    )


# Compatibility for the upstream public helper name.
create_trained_policy = create_policy
