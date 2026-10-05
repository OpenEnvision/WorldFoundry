# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

"""Frame-wise modulation for cached Hunyuan camera inference."""

from einops import rearrange

from worldfoundry.core.nn import DiTFinalLayer


def modulate(x, shift=None, scale=None):
    """modulate by shift and scale

    Args:
        x (torch.Tensor): input tensor.
        shift (torch.Tensor, optional): shift tensor. Defaults to None.
        scale (torch.Tensor, optional): scale tensor. Defaults to None.

    Returns:
        torch.Tensor: the output tensor after modulate.
    """
    if scale is None and shift is None:
        return x
    elif shift is None:
        scale = scale.unsqueeze(0)
        scale = rearrange(scale, "B (N T) C -> (B N) T C", N=x.shape[0])
        latent_length = scale.shape[1]
        token_length = x.shape[1] // latent_length
        scale = scale.repeat_interleave(token_length, dim=1).type_as(x)
        return x * (1 + scale)
    elif scale is None:
        shift = shift.unsqueeze(0)
        shift = rearrange(shift, "B (N T) C -> (B N) T C", N=x.shape[0])
        latent_length = shift.shape[1]
        token_length = x.shape[1] // latent_length
        shift = shift.repeat_interleave(token_length, dim=1).type_as(x)
        return x + shift
    else:
        shift = shift.unsqueeze(0)
        scale = scale.unsqueeze(0)
        shift = rearrange(shift, "B (N T) C -> (B N) T C", N=x.shape[0])
        scale = rearrange(scale, "B (N T) C -> (B N) T C", N=x.shape[0])
        latent_length = shift.shape[1]
        token_length = x.shape[1] // latent_length
        scale = scale.repeat_interleave(token_length, dim=1).type_as(x)
        shift = shift.repeat_interleave(token_length, dim=1).type_as(x)
        return x * (1 + scale) + shift


def apply_gate(x, gate=None, tanh=False):
    """Apply a per-frame residual gate to flattened video tokens.

    Args:
        x (torch.Tensor): input tensor.
        gate (torch.Tensor, optional): gate tensor. Defaults to None.
        tanh (bool, optional): whether to use tanh function. Defaults to False.

    Returns:
        torch.Tensor: the output tensor after apply gate.
    """
    if gate is None:
        return x
    gate = gate.unsqueeze(0)
    gate = rearrange(gate, "B (N T) C -> (B N) T C", N=x.shape[0])
    latent_length = gate.shape[1]
    token_length = x.shape[1] // latent_length
    gate = gate.repeat_interleave(token_length, dim=1).type_as(x)
    if tanh:
        return x * gate.tanh()
    else:
        return x * gate


class FinalLayer(DiTFinalLayer):
    def forward(self, value, condition):
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(value), shift=shift, scale=scale))
