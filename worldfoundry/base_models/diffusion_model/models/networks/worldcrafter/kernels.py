# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
"""Use the existing base-model inference kernels."""
from worldfoundry.base_models.diffusion_model.models.networks.helios.kernels import (
    replace_rmsnorm_with_fp32, replace_all_norms_with_flash_norms, replace_rope_with_flash_rope,
)
