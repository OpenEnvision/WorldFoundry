"""Typed configuration used by pixelSplat inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Type, TypeVar

try:
    from dacite import Config, from_dict
except ImportError:
    from .dacite_compat import Config, from_dict
from omegaconf import DictConfig, OmegaConf

from .dataset import DatasetCfg
from .model.decoder import DecoderCfg
from .model.encoder import EncoderCfg


@dataclass
class ModelCfg:
    decoder: DecoderCfg
    encoder: EncoderCfg


@dataclass
class InferenceCfg:
    checkpoint_path: Path
    output_path: Path
    num_workers: int
    dataset: DatasetCfg
    model: ModelCfg


T = TypeVar("T")


def load_typed_config(cfg: DictConfig, data_class: Type[T]) -> T:
    return from_dict(
        data_class,
        OmegaConf.to_container(cfg),
        config=Config(type_hooks={Path: Path}),
    )


def load_inference_config(cfg: DictConfig) -> InferenceCfg:
    """Extract only inference settings from the upstream-compatible Hydra config."""
    return InferenceCfg(
        checkpoint_path=Path(cfg.checkpointing.load),
        output_path=Path(cfg.test.output_path),
        num_workers=int(cfg.data_loader.test.num_workers),
        dataset=load_typed_config(cfg.dataset, DatasetCfg),
        model=ModelCfg(
            encoder=load_typed_config(cfg.model.encoder, EncoderCfg),
            decoder=load_typed_config(cfg.model.decoder, DecoderCfg),
        ),
    )
