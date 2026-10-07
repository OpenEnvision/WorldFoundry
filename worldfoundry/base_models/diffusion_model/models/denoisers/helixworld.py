"""HelixWorld bindings for the shared LTX denoiser and native module loader."""

from __future__ import annotations

from ..networks.ltx.helixworld import HelixWorldModel
from .ltx import LTXAVTransformerModule, LTXJointDenoiser, build_ltx_joint_denoiser
from .ltx_configurator import LTXModelConfigurator


class HelixWorldModelConfigurator(LTXModelConfigurator):
    MODEL_CLS = HelixWorldModel


class HelixWorldTransformerModule(LTXAVTransformerModule):
    CONFIGURATOR = HelixWorldModelConfigurator


class HelixWorldDenoiser(LTXJointDenoiser):
    @staticmethod
    def _clean_prediction(modality, velocity):
        return (modality.latent.float() - modality.timesteps.float() * velocity.float()).to(modality.latent.dtype)


def build_helixworld_denoiser(context):
    return build_ltx_joint_denoiser(
        context, module_class=HelixWorldTransformerModule, denoiser_class=HelixWorldDenoiser,
    )
