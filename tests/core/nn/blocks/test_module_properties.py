# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""First-tensor dtype probing keeps the None-returning helper distinct from the strict mixin."""

import pytest
import torch
from torch import nn

from worldfoundry.core.nn.blocks.module_properties import (
    ModuleDeviceDtypeMixin,
    get_parameter_dtype,
)

pytestmark = pytest.mark.unit


def test_get_parameter_dtype_prefers_first_parameter_then_first_buffer() -> None:
    module = nn.Module()
    assert get_parameter_dtype(module) is None
    module.register_buffer("indices", torch.arange(2))
    module.register_buffer("values", torch.ones(2))
    assert get_parameter_dtype(module) == torch.int64
    module.register_parameter("weight", nn.Parameter(torch.randn(2, dtype=torch.float16)))
    assert get_parameter_dtype(module) == torch.float16


def test_module_device_dtype_mixin_still_rejects_tensor_free_modules() -> None:
    class TensorModule(ModuleDeviceDtypeMixin, nn.Module):
        pass

    with pytest.raises(RuntimeError, match="does not own a parameter or buffer"):
        _ = TensorModule().dtype
