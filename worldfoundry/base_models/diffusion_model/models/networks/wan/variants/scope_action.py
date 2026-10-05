"""SCOPE (Wan 2.2 TI2V): mouse / keyboard action residuals on native Wan blocks.

:class:`ScopeActionModule` windows raw action streams onto latent frames,
then adds a mouse self-attn residual and a keyboard cross-attn residual
between the host block's text cross-attn and FFN.
:class:`ScopeActionWanModel` installs those blocks and passes grid /
action kwargs through :meth:`WanModel.block_forward_kwargs`.

Adds action conditioning and per-token timesteps.  No VACE, camera
ControlNet, causal, or linear-attention processors.
"""

from __future__ import annotations

from typing import Any

import torch
from einops import rearrange, repeat
from torch import nn

from worldfoundry.core.attention import (
    apply_complex_rotary_embedding,
    complex_rotary_frequencies,
    packed_sequence_attention,
)
from worldfoundry.core.nn import RMSNorm, scale_shift

from ..model import DiTBlock, SelfAttention, WanModel


class ScopeActionModule(nn.Module):
    """Fuse mouse/joystick and keyboard/button sequences into Wan tokens."""

    def __init__(
        self,
        mouse_dim_in: int,
        keyboard_dim_in: int,
        dim: int,
        num_heads: int,
        vae_time_compression_ratio: int = 4,
        windows_size: int = 1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.vae_time_compression_ratio = vae_time_compression_ratio
        self.windows_size = windows_size
        self.num_heads = num_heads
        window = vae_time_compression_ratio * windows_size

        self.mouse_mlp = nn.Sequential(
            nn.Linear(mouse_dim_in * window + dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.mouse_attn = SelfAttention(dim, num_heads)

        keyboard_hidden = dim // window
        self.keyboard_embed = nn.Sequential(
            nn.Linear(keyboard_dim_in, keyboard_hidden),
            nn.SiLU(),
            nn.Linear(keyboard_hidden, keyboard_hidden),
        )
        self.keyboard_q_proj = nn.Linear(dim, dim, bias=False)
        self.keyboard_kv_proj = nn.Linear(dim, dim * 2, bias=False)
        self.keyboard_o_proj = nn.Linear(dim, dim, bias=False)
        self.key_attn_q_norm = RMSNorm(dim, eps=eps)
        self.key_attn_k_norm = RMSNorm(dim, eps=eps)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.mouse_attn.o.weight)
        nn.init.zeros_(self.mouse_attn.o.bias)
        nn.init.zeros_(self.keyboard_o_proj.weight)

    def _windows(self, action: torch.Tensor, latent_frames: int) -> torch.Tensor:
        """Slice a raw action stream into one window per latent frame."""
        window = self.vae_time_compression_ratio * self.windows_size
        prefix = action[:, :1].expand(-1, window, -1)
        padded = torch.cat((prefix, action), dim=1)
        required = (latent_frames - 1) * self.vae_time_compression_ratio + window
        if padded.shape[1] < required:
            suffix = padded[:, -1:].expand(-1, required - padded.shape[1], -1)
            padded = torch.cat((padded, suffix), dim=1)
        return torch.stack(
            [
                padded[
                    :,
                    index * self.vae_time_compression_ratio :
                    index * self.vae_time_compression_ratio + window,
                ]
                for index in range(latent_frames)
            ],
            dim=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        frames: int,
        height: int,
        width: int,
        freqs: torch.Tensor,
        mouse_action: torch.Tensor | None = None,
        keyboard_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add mouse and/or keyboard residuals; tokens stay ``[B, F*H*W, C]``."""
        batch, sequence, channels = x.shape
        spatial = height * width
        if sequence != frames * spatial:
            raise ValueError("SCOPE token sequence does not match its latent grid")
        hidden = rearrange(x, "b (f s) c -> (b s) f c", f=frames, s=spatial)

        if mouse_action is not None:
            mouse = self._windows(mouse_action, frames)
            mouse = repeat(mouse, "b f p c -> (b s) f (p c)", s=spatial)
            hidden = hidden + self.mouse_attn(
                self.mouse_mlp(torch.cat((hidden, mouse), dim=-1)),
                freqs,
            )

        if keyboard_action is not None:
            keyboard = self._windows(self.keyboard_embed(keyboard_action), frames)
            keyboard = rearrange(keyboard, "b f p d -> b f (p d)")
            query = self.key_attn_q_norm(self.keyboard_q_proj(hidden))
            key, value = self.keyboard_kv_proj(keyboard).chunk(2, dim=-1)
            key = self.key_attn_k_norm(key)
            key = repeat(key, "b f d -> (b s) f d", s=spatial)
            value = repeat(value, "b f d -> (b s) f d", s=spatial)
            query = apply_complex_rotary_embedding(query, freqs, self.num_heads)
            key = apply_complex_rotary_embedding(key, freqs, self.num_heads)
            hidden = hidden + self.keyboard_o_proj(
                packed_sequence_attention(query, key, value, num_heads=self.num_heads)
            )

        return rearrange(hidden, "(b s) f c -> b (f s) c", b=batch, s=spatial)


class ScopeActionBlock(DiTBlock):
    """Canonical Wan block with the SCOPE action residual inserted."""

    def __init__(
        self,
        has_image_input: bool,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
        *,
        action_config: dict[str, Any],
    ) -> None:
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps)
        self.action_attn = ScopeActionModule(**action_config)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        *,
        frames: int,
        height: int,
        width: int,
        action_freqs: torch.Tensor,
        mouse_action: torch.Tensor | None = None,
        keyboard_action: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Wan AdaLN self-attn → text cross-attn → SCOPE action → gated FFN."""
        self_attention_kwargs = {
            key: kwargs[key]
            for key in (
                "_worldfoundry_rope_precision",
                "_worldfoundry_rope_grid",
                "_worldfoundry_rope_table",
                "_worldfoundry_sparse_grid",
            )
            if key in kwargs
        }
        sequence_timestep = t_mod.ndim == 4
        chunk_dim = 2 if sequence_timestep else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if sequence_timestep:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                value.squeeze(2)
                for value in (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
            )
        x = self.gate(
            x,
            gate_msa,
            self.self_attn(
                scale_shift(self.norm1(x), shift_msa, scale_msa),
                freqs,
                **self_attention_kwargs,
            ),
        )
        x = x + self.cross_attn(self.norm3(x), context)
        x = self.action_attn(
            x,
            frames=frames,
            height=height,
            width=width,
            freqs=action_freqs,
            mouse_action=mouse_action,
            keyboard_action=keyboard_action,
        )
        return self.gate(
            x,
            gate_mlp,
            self.ffn(scale_shift(self.norm2(x), shift_mlp, scale_mlp)),
        )


class ScopeActionWanModel(WanModel):
    """Wan2.2 TI2V transformer with SCOPE action conditioning."""

    def __init__(
        self,
        *args: Any,
        enable_action: bool = True,
        action_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not enable_action:
            raise ValueError("ScopeActionWanModel requires action conditioning")
        if action_config is None:
            raise ValueError("ScopeActionWanModel requires action_config")
        kwargs["per_token_timestep"] = bool(kwargs.get("seperated_timestep", True))
        super().__init__(
            *args,
            block_class=ScopeActionBlock,
            block_kwargs={"action_config": dict(action_config)},
            **kwargs,
        )
        head_dim = self.dim // int(action_config["num_heads"])
        self.freqs_mouse = complex_rotary_frequencies(head_dim, end=100)

    def block_forward_kwargs(
        self,
        grid_size: tuple[int, int, int],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Forward latent grid plus mouse/keyboard tensors into every SCOPE block."""
        frames, height, width = grid_size
        mouse_action = kwargs.get("mouse_action")
        keyboard_action = kwargs.get("keyboard_action")
        action = mouse_action if mouse_action is not None else keyboard_action
        action_freqs = self.freqs_mouse[:frames].view(frames, 1, -1)
        if action is not None:
            action_freqs = action_freqs.to(action.device)
        return {
            "frames": frames,
            "height": height,
            "width": width,
            "action_freqs": action_freqs,
            "mouse_action": mouse_action,
            "keyboard_action": keyboard_action,
        }


__all__ = ["ScopeActionBlock", "ScopeActionModule", "ScopeActionWanModel"]
