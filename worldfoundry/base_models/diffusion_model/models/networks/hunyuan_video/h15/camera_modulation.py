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

from worldfoundry.core.nn import DiTFinalLayer


def modulate(x, shift=None, scale=None):
    """Modulate ``[B, S, C]`` tokens with per-frame ``[B*T, C]`` shift/scale via zero-copy broadcast."""
    if scale is None and shift is None:
        return x
    batch, seq_len, hidden = x.shape
    ref = scale if scale is not None else shift
    latent_length = ref.shape[0] // batch
    token_length = seq_len // latent_length
    if latent_length == seq_len:
        scale_v = None if scale is None else scale.reshape(batch, seq_len, hidden).type_as(x)
        shift_v = None if shift is None else shift.reshape(batch, seq_len, hidden).type_as(x)
        if shift_v is None:
            return x * (1 + scale_v)
        if scale_v is None:
            return x + shift_v
        return x * (1 + scale_v) + shift_v
    if latent_length == 1:
        scale_v = None if scale is None else scale.reshape(batch, 1, hidden).type_as(x)
        shift_v = None if shift is None else shift.reshape(batch, 1, hidden).type_as(x)
        if shift_v is None:
            return x * (1 + scale_v)
        if scale_v is None:
            return x + shift_v
        return x * (1 + scale_v) + shift_v
    if latent_length * token_length == seq_len:
        x_v = x.reshape(batch, latent_length, token_length, hidden)
        scale_v = None if scale is None else scale.reshape(batch, latent_length, 1, hidden).type_as(x)
        shift_v = None if shift is None else shift.reshape(batch, latent_length, 1, hidden).type_as(x)
        if shift_v is None:
            out = x_v * (1 + scale_v)
        elif scale_v is None:
            out = x_v + shift_v
        else:
            out = x_v * (1 + scale_v) + shift_v
        return out.reshape(batch, seq_len, hidden)
    raise ValueError(f"Sequence length {seq_len} is not divisible by modulation length {latent_length}")


def apply_gate(x, gate=None, tanh=False):
    """Apply a per-frame ``[B*T, C]`` residual gate to ``[B, S, C]`` tokens without repeat_interleave."""
    if gate is None:
        return x
    batch, seq_len, hidden = x.shape
    latent_length = gate.shape[0] // batch
    token_length = seq_len // latent_length
    g = gate.tanh() if tanh else gate
    if latent_length == seq_len:
        return x * g.reshape(batch, seq_len, hidden).type_as(x)
    if latent_length == 1:
        return x * g.reshape(batch, 1, hidden).type_as(x)
    if latent_length * token_length == seq_len:
        return (
            x.reshape(batch, latent_length, token_length, hidden)
            * g.reshape(batch, latent_length, 1, hidden).type_as(x)
        ).reshape(batch, seq_len, hidden)
    raise ValueError(f"Sequence length {seq_len} is not divisible by modulation length {latent_length}")


class FinalLayer(DiTFinalLayer):
    def forward(self, value, condition):
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(value), shift=shift, scale=scale))
