from __future__ import annotations

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.dreamx_world.transformer import (
    WanTransformer3DModel,
)
from worldfoundry.core.nn import ModuleDeviceDtypeMixin


def test_dreamx_transformer_exposes_diffusers_device_dtype_contract() -> None:
    assert issubclass(WanTransformer3DModel, ModuleDeviceDtypeMixin)
