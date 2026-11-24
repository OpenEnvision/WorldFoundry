# Copyright 2025 The Emu team, BAAI and The HuggingFace Inc. team. All rights reserved.
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
"""Emu3p5VisionVQ model configuration"""

from transformers.configuration_utils import PretrainedConfig


class Emu3p5VisionVQConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Emu3p5VisionVQ`]. It is used to instantiate an video movq
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a configuration to the VQ model presented in Emu3p5 paper.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        codebook_size (`int`, *optional*, defaults to 32768):
            Codebook size of the VQ model.
        embed_dim (`int`, *optional*, defaults to 4):
            Dimension of the quantized vector in codebook.
        z_channels (`int`, *optional*, defaults to 4):
            Dimension of the output channel of encoder and the input channel of decoder
        double_z (`bool`, *optional*, defaults to False):
            Whether double the output dim of the encoder.
        in_channels (`int`, *optional*, defaults to 3):
            Input channel of encoder.
        out_channels (`int`, *optional*, defaults to 3):
            Output channel of decoder.
        temporal_downsample_factor (`int`, *optional*, defaults to 4):
            Temporal downsample factor.
        ch (`int`, *optional*, defaults to 256):
            Basic channel number of the intermediate blocks.
        ch_mult (`List[int]`, *optional*, defaults to `[1, 2, 2, 4]`):
            Channel scaling factor of the intermediate blocks.
        num_res_blocks (`int`, *optional*, defaults to 2):
            Residual block number in each stage.
        attn_resolutions (`List[int]`, *optional*, defaults to 3):
            Stage indices to apply attention.
        dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability.
    ```python
    >>> from configuration_emu3p5visionvq import Emu3VisionVQConfig
    >>> from modeling_emu3p5visionvq import Emu3VisionVQ
    >>> # Initializing a video VQ model of Emu3 configuration
    >>> configuration = Emu3VisionVQConfig()
    >>> # Initializing a model from the Emu3 VQ model style configuration
    >>> model = Emu3VisionVQModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "Emu3p5VisionVQ"

    def __init__(
        self,
        double_z: bool = False,
        z_channels: int = 256,
        resolution: int = 256,
        in_channels: int = 3,
        out_ch: int = 3,
        ch: int = 256,
        ch_mult: list[int] | None = None,
        num_res_blocks: int = 4,
        attn_resolutions: list[int] | None = None,
        dropout: float = 0.0,
        codebook_size: int = 131072,
        embed_dim: int = 256,
        **kwargs,
    ):
        if attn_resolutions is None:
            attn_resolutions = [16]
        if ch_mult is None:
            ch_mult = [1, 1, 2, 2, 4]
        super().__init__(**kwargs)

        self.double_z = double_z
        self.z_channels = z_channels
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_ch = out_ch
        self.ch = ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout

        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
