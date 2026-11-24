"""DurationHead: predicts shot duration from Connector token outputs."""

from worldfoundry.base_models.diffusion_model.models.networks.ltx.duration_head import AttentionPooler, DurationHead

from .model_configurator import DURATION_HEAD_KEY_OPS, DurationHeadConfigurator

__all__ = [
    "DURATION_HEAD_KEY_OPS",
    "AttentionPooler",
    "DurationHead",
    "DurationHeadConfigurator",
]
