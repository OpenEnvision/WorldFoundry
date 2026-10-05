"""Compatibility import for Evoke checkpoint model_index.json."""

from diffusers import ModelMixin

from worldfoundry.base_models.diffusion_model.models.networks.evoke.model import EvokeTransformer3DModel

__all__ = ["EvokeTransformer3DModel", "ModelMixin"]
