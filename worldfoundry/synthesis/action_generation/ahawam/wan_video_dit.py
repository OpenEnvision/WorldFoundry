from typing import Any

import torch
import torch.nn as nn
from einops import rearrange

from worldfoundry.base_models.diffusion_model.models.networks.gamma_world.multiagent_rope import (
    precompute_freqs_cis_3d,
    sinusoidal_embedding_1d,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.wam import (
    NativeDtypeHead,
    WamDiTBlock,
)
from worldfoundry.core.attention import scaled_dot_product_attention


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    ctx_mask: torch.Tensor | None = None,
    compatibility_mode=True,
):
    if compatibility_mode:
        bsz, q_seq_len, _ = q.shape
        k_seq_len = k.shape[1]
        head_dim = q.shape[2] // num_heads
        q = q.view(bsz, q_seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, k_seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, k_seq_len, num_heads, head_dim).transpose(1, 2)
        if ctx_mask is not None and ctx_mask.ndim == 3:
            ctx_mask = ctx_mask.unsqueeze(1)
        x = scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
        x = x.transpose(1, 2).reshape(bsz, q_seq_len, -1)
        return x
    else:
        raise NotImplementedError(
            "Only compatibility mode is implemented for flash attention. Please set compatibility_mode=True."
        )


def _rope_apply_real(x, cos_f, sin_f, num_heads):
    bsz, seq_len, _ = x.shape
    x = x.view(bsz, seq_len, num_heads, -1)
    x_pairs = x.to(torch.float32).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    x1, x2 = x_pairs.unbind(-1)
    x_out = torch.stack(
        [
            x1 * cos_f - x2 * sin_f,
            x1 * sin_f + x2 * cos_f,
        ],
        dim=-1,
    ).flatten(2)
    return x_out.to(x.dtype)


def rope_apply(x, freqs, num_heads):
    # Decompose complex freqs in eager mode to avoid TorchInductor complex operator warning
    return _rope_apply_real(x, freqs.real, freqs.imag, num_heads)


class WanVideoDiT(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: tuple[int, int, int],
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        rope_max_length: int,
        rope_theta: float,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = False,
        require_clip_embedding: bool = False,
        fuse_vae_embedding_in_latents: bool = True,
        action_conditioned: bool = False,
        action_dim: int = 7,
        action_group_causal_mask_mode="causal",
        video_attention_mask_mode: str = "bidirectional",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.rope_max_length = int(rope_max_length)
        self.rope_theta = float(rope_theta)
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.video_attention_mask_mode = str(video_attention_mask_mode)

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")

        self.action_conditioned = action_conditioned
        self.action_dim = action_dim
        if has_image_input or require_clip_embedding:
            raise ValueError("AHA-WAM uses latent-only video conditioning")
        if require_vae_embedding or not fuse_vae_embedding_in_latents:
            raise ValueError("AHA-WAM requires VAE embeddings fused directly into latents")

        self.patch_embedding = nn.Conv3d(in_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.GELU(approximate="tanh"), nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [
                WamDiTBlock(
                    hidden_dim, attn_head_dim, num_heads, ffn_dim, eps, attention_fn=flash_attention, rope_fn=rope_apply
                )
                for _ in range(num_layers)
            ]
        )
        self.head = NativeDtypeHead(hidden_dim, out_dim, patch_size, eps)
        self.freqs = precompute_freqs_cis_3d(
            attn_head_dim,
            end=self.rope_max_length,
            theta=self.rope_theta,
        )
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, hidden_dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        self.control_adapter = None

        if self.action_conditioned:
            self.action_embedding = nn.Linear(action_dim, hidden_dim)
            self.action_group_causal_mask_mode = action_group_causal_mask_mode

    def patchify(self, x: torch.Tensor, control_camera_latents_input: torch.Tensor | None = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera, strict=False)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def _validate_forward_inputs(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
        action: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"`latents` must be 5D [B, C, T, H, W], got shape {tuple(x.shape)}")
        num_latent_frames = x.shape[2]
        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}")
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        if self.action_conditioned:
            allow_text_only_single_frame = num_latent_frames == 1 and action is None
            if not allow_text_only_single_frame:
                assert action is not None, "Action input is required for action-conditioned model."
                if action.ndim != 3:
                    raise ValueError(
                        f"`action` must be 3D [B, action_horizon, action_dim], got shape {tuple(action.shape)}"
                    )
                if action.shape[2] != self.action_dim:
                    raise ValueError(f"`action` last dimension must be {self.action_dim}, got {action.shape[2]}")
                if num_latent_frames <= 1:
                    raise ValueError(f"video length must be > 1 for action-conditioned model, got {num_latent_frames}")
                if action.shape[1] % (num_latent_frames - 1) != 0:
                    raise ValueError(
                        f"action horizon must be divisible by (num_latent_frames - 1), got action_horizon={action.shape[1]}"
                    )
        if context_mask is None:
            context_mask = torch.ones((context.shape[0], context.shape[1]), dtype=torch.bool, device=context.device)
        else:
            if context_mask.ndim != 2:
                raise ValueError(f"`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != context.shape[0] or context_mask.shape[1] != context.shape[1]:
                raise ValueError(
                    f"`context_mask` shape must match `context` shape [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
                )

        batch_size = x.shape[0]
        if batch_size != context.shape[0]:
            if batch_size == 1:
                x = x.expand(context.shape[0], -1, -1, -1, -1)
                batch_size = context.shape[0]
            else:
                raise ValueError(f"Batch mismatch between latents and context: {batch_size} vs {context.shape[0]}.")

        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}")
        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        return x, timestep, context_mask

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if video_seq_len <= 0:
            raise ValueError(f"`video_seq_len` must be positive, got {video_seq_len}")
        if video_tokens_per_frame <= 0:
            raise ValueError(f"`video_tokens_per_frame` must be positive, got {video_tokens_per_frame}")

        if self.video_attention_mask_mode == "bidirectional":
            return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

        if self.video_attention_mask_mode == "per_frame_causal":
            if video_seq_len % video_tokens_per_frame != 0:
                raise ValueError(
                    "`video_seq_len` must be divisible by `video_tokens_per_frame` in `per_frame_causal` mode, "
                    f"got {video_seq_len} and {video_tokens_per_frame}"
                )
            num_video_frames = video_seq_len // video_tokens_per_frame
            frame_causal = torch.tril(torch.ones((num_video_frames, num_video_frames), dtype=torch.bool, device=device))
            return frame_causal.repeat_interleave(video_tokens_per_frame, dim=0).repeat_interleave(
                video_tokens_per_frame, dim=1
            )

        if self.video_attention_mask_mode == "first_frame_causal":
            video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
            video_mask[:first_frame_tokens, first_frame_tokens:] = False
            return video_mask

        raise ValueError(f"Unsupported video attention mask mode: {self.video_attention_mask_mode}")

    def pre_dit(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        temporal_position_offset: int = 0,
        temporal_position_ids: torch.Tensor | None = None,
        fuse_vae_embedding_in_latents: bool = False,
        control_camera_latents_input: torch.Tensor | None = None,
        clean_prefix_frames: int = 1,
    ) -> dict[str, Any]:
        x, timestep, context_mask = self._validate_forward_inputs(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
        )

        batch_size = x.shape[0]
        patch_t = int(self.patch_size[0])
        patch_h = int(self.patch_size[1])
        patch_w = int(self.patch_size[2])
        if x.shape[2] % patch_t != 0 or x.shape[3] % patch_h != 0 or x.shape[4] % patch_w != 0:
            raise ValueError(
                "Latent shape must be divisible by DiT patch size, "
                f"got TxHxW={tuple(x.shape[2:])}, patch={tuple(self.patch_size)}"
            )

        x = self.patchify(x, control_camera_latents_input=control_camera_latents_input)
        num_frames, grid_h, grid_w = x.shape[2:]
        if self.seperated_timestep and fuse_vae_embedding_in_latents:
            if not hasattr(self, "patch_size") or len(self.patch_size) < 3:
                raise ValueError(f"Invalid dit.patch_size: {getattr(self, 'patch_size', None)}")

            token_timesteps = torch.ones(
                (batch_size, num_frames, grid_h * grid_w),
                dtype=timestep.dtype,
                device=timestep.device,
            ) * timestep.view(batch_size, 1, 1)
            clean_prefix_frames = max(0, min(int(clean_prefix_frames), num_frames))
            if clean_prefix_frames > 0:
                token_timesteps[:, :clean_prefix_frames, :] = 0
            token_timesteps = token_timesteps.reshape(batch_size, -1)
            token_t_emb = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
            t = self.time_embedding(token_t_emb).reshape(batch_size, -1, self.hidden_dim)
            t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))
        else:
            raise NotImplementedError("Only support seperated_timestep with fuse_vae_embedding_in_latents for now.")

        if temporal_position_ids is None:
            frame_positions = torch.arange(num_frames, device=x.device) + int(temporal_position_offset)
        else:
            frame_positions = torch.as_tensor(temporal_position_ids, device=x.device)
            if frame_positions.ndim != 1 or frame_positions.numel() != num_frames:
                raise ValueError(
                    f"`temporal_position_ids` must have one entry per latent frame ({num_frames}), "
                    f"got shape {tuple(frame_positions.shape)}"
                )
            if frame_positions.dtype not in (torch.int32, torch.int64):
                raise ValueError("`temporal_position_ids` must contain integer frame indices")
        if torch.any(frame_positions < 0) or torch.any(frame_positions >= self.rope_max_length):
            raise ValueError(f"Temporal RoPE positions must be in [0, {self.rope_max_length})")
        if grid_h > self.rope_max_length or grid_w > self.rope_max_length:
            raise ValueError("Latent spatial grid exceeds the RoPE cache")

        f_freqs, h_freqs, w_freqs = self.freqs
        f_rope = f_freqs[frame_positions.long()].view(num_frames, 1, 1, -1)
        h_rope = h_freqs[:grid_h].view(1, grid_h, 1, -1)
        w_rope = w_freqs[:grid_w].view(1, 1, grid_w, -1)
        freqs = (
            torch.cat(
                (
                    f_rope.expand(-1, grid_h, grid_w, -1),
                    h_rope.expand(num_frames, -1, grid_w, -1),
                    w_rope.expand(num_frames, grid_h, -1, -1),
                ),
                dim=-1,
            )
            .reshape(num_frames * grid_h * grid_w, 1, -1)
            .to(x.device)
        )
        tokens = rearrange(x, "b c f h w -> b (f h w) c")
        return {
            "tokens": tokens,
            "context": self.text_embedding(context),
            "context_mask": context_mask,
            "t": t,
            "t_mod": t_mod,
            "freqs": freqs,
            "meta": {
                "grid_size": (num_frames, grid_h, grid_w),
                "tokens_per_frame": grid_h * grid_w,
            },
        }

    def post_dit(self, x_tokens: torch.Tensor, pre_state: dict[str, Any]) -> torch.Tensor:
        f, h, w = pre_state["meta"]["grid_size"]
        x = self.head(x_tokens, pre_state["t"])
        x = self.unpatchify(x, (f, h, w))
        return x

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        pre_state = self.pre_dit(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        x_tokens = pre_state["tokens"]
        context_emb = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_attn_mask = pre_state["context_mask"]
        self_attn_mask = (
            self.build_video_to_video_mask(
                video_seq_len=x_tokens.shape[1],
                video_tokens_per_frame=int(pre_state["meta"]["tokens_per_frame"]),
                device=x_tokens.device,
            )
            if self.video_attention_mask_mode != "bidirectional"
            else None
        )  # special rule for faster speed

        for block in self.blocks:
            x_tokens = block(
                x_tokens, context_emb, t_mod, freqs, context_mask=context_attn_mask, self_attn_mask=self_attn_mask
            )

        return self.post_dit(x_tokens, pre_state)
