# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared Cosmos Predict1 Adams–Bashforth multistep ODE helpers.

:func:`order2_fn` is the 2nd-order multistep update from the
Predict1 residual-sampler paper (Adams–Bashforth): it reuses the
previous x0 prediction when available, otherwise falls back to a
regularized Euler step.

:func:`get_multi_step_fn` looks up ``"2ab"``.  Runge–Kutta single-step
methods live in :mod:`.runge_kutta`; this file is the multistep
registry only.
"""

from typing import Callable, List, Tuple

import torch

from worldfoundry.base_models.diffusion_model.schedulers.cosmos.runge_kutta import (
    reg_x0_euler_step,
    res_x0_rk2_step,
)


def order2_fn(
    x_s: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x0_s: torch.Tensor, x0_preds: torch.Tensor
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    impl the second order multistep method in https://arxiv.org/pdf/2308.02157
    Adams Bashforth approach!
    """
    if x0_preds:
        x0_s1, s1 = x0_preds[0]
        x_t = res_x0_rk2_step(x_s, t, s, x0_s, s1, x0_s1)
    else:
        x_t = reg_x0_euler_step(x_s, s, t, x0_s)[0]
    return x_t, [(x0_s, s)]


MULTISTEP_FNs = {
    "2ab": order2_fn,
}


def get_multi_step_fn(name: str) -> Callable:
    """Get multi step fn.

    Args:
        name: The name.

    Returns:
        The return value.
    """
    if name in MULTISTEP_FNs:
        return MULTISTEP_FNs[name]
    methods = "\n\t".join(MULTISTEP_FNs.keys())
    raise RuntimeError("Only support multistep method\n" + methods)


def is_multi_step_fn_supported(name: str) -> bool:
    """Check if the multistep method is supported."""
    return name in MULTISTEP_FNs
