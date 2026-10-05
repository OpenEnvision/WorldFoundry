# Copyright 2024 MIT Han Lab
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
#
# SPDX-License-Identifier: Apache-2.0

"""Triton-fused LiteMLA and MBConvPreGLU used by SANA ``triton_linear`` blocks.

:class:`TritonLiteMLA` fuses QKV+ReLU, ``VK`` accumulation, and the
normalized ``Q @ VK`` projection of linear attention.  :class:`TritonMBConvPreGLU`
fuses the pre-GLU linear and the depthwise conv of the mobile FFN.
PyTorch reference modules (:class:`LiteMLA`, :class:`MBConvPreGLU`) live
alongside for CPU / non-Triton fallbacks.
"""

from .triton_lite_mla import *
from .triton_lite_mla_fwd import *
from .triton_mb_conv_pre_glu import *

# from .flash_attn import *
