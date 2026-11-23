# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Memory-efficient MMD implementation in JAX."""

import torch

# The bandwidth parameter for the Gaussian RBF kernel. See the paper for more
# details.
_SIGMA = 10
# The following is used to make the metric more human readable. See the paper
# for more details.
_SCALE = 1000


def mmd(x, y):
    """Memory-efficient MMD with Gaussian RBF kernel (CMMD paper)."""
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)

    x_sqnorms = torch.sum(x * x, dim=1)
    y_sqnorms = torch.sum(y * y, dim=1)

    gamma = 1 / (2 * _SIGMA**2)
    k_xx = torch.mean(torch.exp(-gamma * (-2 * torch.matmul(x, x.T) + x_sqnorms[:, None] + x_sqnorms[None, :])))
    k_xy = torch.mean(torch.exp(-gamma * (-2 * torch.matmul(x, y.T) + x_sqnorms[:, None] + y_sqnorms[None, :])))
    k_yy = torch.mean(torch.exp(-gamma * (-2 * torch.matmul(y, y.T) + y_sqnorms[:, None] + y_sqnorms[None, :])))

    return _SCALE * (k_xx + k_yy - 2 * k_xy)
