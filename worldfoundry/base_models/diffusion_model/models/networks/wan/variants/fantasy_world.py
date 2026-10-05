"""FantasyWorld: IRG bidirectional fusion + Wan 2.1 camera adapter.

Only FantasyWorld-specific roles live here (IRG blocks, camera pose
encoder, fusion model).  The host DiT remains canonical native Wan.

Adds camera / IRG fusion.  No VACE trunk, causal cache, or linear attn.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from worldfoundry.core.attention import (
    apply_complex_rotary_embedding,
    complex_rotary_frequencies_3d,
    scaled_dot_product_attention,
)
from worldfoundry.core.nn import sinusoidal_embedding_1d


def build_frequencies_with_frame_tokens(
    frequencies_3d: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    frames: int,
    height: int,
    width: int,
    *,
    extra_tokens_per_frame: int,
    device: torch.device,
) -> torch.Tensor:
    """Build 3D RoPE values for VGGT frame tokens followed by patch tokens."""

    temporal, vertical, horizontal = frequencies_3d
    temporal_grid = temporal[:frames].view(frames, 1, 1, -1).expand(frames, height, width, -1)
    vertical_grid = vertical[:height].view(1, height, 1, -1).expand(frames, height, width, -1)
    horizontal_grid = horizontal[:width].view(1, 1, width, -1).expand(frames, height, width, -1)
    patch = torch.cat((temporal_grid, vertical_grid, horizontal_grid), dim=-1)
    patch = patch.reshape(frames, height * width, -1)
    extra = torch.ones(
        frames,
        extra_tokens_per_frame,
        patch.shape[-1],
        dtype=patch.dtype,
        device=patch.device,
    )
    return torch.cat((extra, patch), dim=1).reshape(-1, 1, patch.shape[-1]).to(device)


class BiMultiHeadAttention(nn.Module):
    """Exact two-way SDPA used by the released FantasyWorld IRG blocks."""

    def __init__(
        self,
        modality_one_dim: int,
        modality_two_dim: int,
        embed_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("FantasyWorld fusion width must be divisible by its head count")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.m1_dim = modality_one_dim
        self.m2_dim = modality_two_dim

        # Attribute names are the public checkpoint schema.
        self.m1_proj = nn.Linear(modality_one_dim, embed_dim)
        self.m2_proj = nn.Linear(modality_two_dim, embed_dim)
        self.values_m1_proj = nn.Linear(modality_one_dim, embed_dim)
        self.values_m2_proj = nn.Linear(modality_two_dim, embed_dim)
        self.out_m1_proj = nn.Linear(embed_dim, modality_one_dim)
        self.out_m2_proj = nn.Linear(embed_dim, modality_two_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for projection in (
            self.m1_proj,
            self.m2_proj,
            self.values_m1_proj,
            self.values_m2_proj,
            self.out_m1_proj,
            self.out_m2_proj,
        ):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        return value.view(value.shape[0], value.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        modality_one: torch.Tensor,
        modality_two: torch.Tensor,
        *,
        frequencies_one: torch.Tensor | None = None,
        frequencies_two: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.m1_proj(modality_one)
        key = self.m2_proj(modality_two)
        if frequencies_one is not None:
            if frequencies_two is None:
                raise ValueError("both FantasyWorld fusion RoPE tables are required")
            query = apply_complex_rotary_embedding(query, frequencies_one, self.num_heads)
            key = apply_complex_rotary_embedding(key, frequencies_two, self.num_heads)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value_one = self._split_heads(self.values_m1_proj(modality_one))
        value_two = self._split_heads(self.values_m2_proj(modality_two))
        update_one = scaled_dot_product_attention(query, key, value_two, dropout_p=0.0)
        update_two = scaled_dot_product_attention(key, query, value_one, dropout_p=0.0)

        update_one = update_one.transpose(1, 2).reshape(
            modality_one.shape[0], modality_one.shape[1], self.embed_dim
        )
        update_two = update_two.transpose(1, 2).reshape(
            modality_two.shape[0], modality_two.shape[1], self.embed_dim
        )
        return self.out_m1_proj(update_one), self.out_m2_proj(update_two)


class CrossModalityBiAttentionBlock(nn.Module):
    """Residual bidirectional Wan/VGGT fusion for one paired block."""

    def __init__(
        self,
        modality_one_dim: int,
        modality_two_dim: int,
        hidden_size: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.attn_norm_m1 = nn.LayerNorm(modality_one_dim, eps=1e-6, elementwise_affine=False)
        self.attn_norm_m2 = nn.LayerNorm(modality_two_dim, eps=1e-6, elementwise_affine=False)
        self.cross_attn = BiMultiHeadAttention(
            modality_one_dim,
            modality_two_dim,
            hidden_size,
            num_heads,
        )
        self.gamma_m1 = nn.Parameter(torch.zeros(modality_one_dim))
        self.gamma_m2 = nn.Parameter(torch.zeros(modality_two_dim))
        self.drop_path = nn.Identity()

    def forward(
        self,
        values: tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
        *,
        frequencies_one: torch.Tensor,
        frequencies_two: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        modality_one, modality_two = values
        update_one, update_two = self.cross_attn(
            self.attn_norm_m1(modality_one),
            self.attn_norm_m2(modality_two),
            frequencies_one=frequencies_one,
            frequencies_two=frequencies_two,
        )
        return (
            modality_one + self.drop_path(self.gamma_m1 * update_one),
            modality_two + self.drop_path(self.gamma_m2 * update_two),
        )


class IRGBlock(nn.Module):
    """Pair one canonical Wan block with one Fantasy VGGT global block."""

    def __init__(
        self,
        x_dit_block: nn.Module,
        x_agg_block: nn.Module,
        *,
        modality_one_dim: int,
        modality_two_dim: int,
        hidden_size: int = 1152,
        num_heads: int = 12,
    ) -> None:
        super().__init__()
        self.x_dit = x_dit_block
        self.x_agg = x_agg_block
        self.bicross_attention = CrossModalityBiAttentionBlock(
            modality_one_dim,
            modality_two_dim,
            hidden_size,
            num_heads,
        )

    def forward(
        self,
        x_dit: torch.Tensor,
        x_agg: torch.Tensor,
        *,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        frequencies_dit: torch.Tensor,
        fusion_frequencies_dit: torch.Tensor,
        fusion_frequencies_agg: torch.Tensor,
        pos: torch.Tensor,
        e0: torch.Tensor | None,
        unconditioned_geometry: bool = False,
        **wan_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        _, patches_per_frame, aggregate_dim = x_agg.shape
        dit_partial, dit_modifiers = self.x_dit.forward_partial(
            x_dit,
            context,
            t_mod,
            frequencies_dit,
            **wan_kwargs,
        )
        pos = rearrange(pos, "(b s) p d -> b (s p) d", b=dit_partial.shape[0])
        x_agg = rearrange(x_agg, "(b s) p d -> b (s p) d", b=dit_partial.shape[0])
        batch = x_agg.shape[0]
        agg_partial, agg_modifiers = self.x_agg(
            x_agg,
            pos=pos,
            e0=e0,
            return_partial=True,
        )
        if unconditioned_geometry:
            dit_fused, agg_fused = dit_partial, agg_partial
        else:
            dit_fused, agg_fused = self.bicross_attention(
                (dit_partial, agg_partial),
                frequencies_one=fusion_frequencies_dit,
                frequencies_two=fusion_frequencies_agg,
            )
        dit_output = self.x_dit.forward_remaining(dit_fused, *dit_modifiers)
        agg_output = self.x_agg(
            agg_fused,
            run_remaining=True,
            modifiers=agg_modifiers,
        )
        return (
            dit_output,
            agg_output,
            [agg_output.view(batch, -1, patches_per_frame, aggregate_dim)],
        )


class GroupLinearDualK(nn.Module):
    """Checkpoint-shaped camera feature and Wan feature projections."""

    def __init__(self, context_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.group1 = nn.Linear(context_dim, context_dim)
        intermediate_dim = min(hidden_dim, context_dim) // 2
        self.group2 = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, context_dim),
        )

    def forward(self, camera: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.group1(camera), self.group2(hidden)


class GroupLinearDualV(nn.Module):
    """Zero-initialized camera shift projection used by Wan2.1 FantasyWorld."""

    def __init__(self, context_dim: int, hidden_dim: int) -> None:
        super().__init__()
        reduced_dim = context_dim // 5
        self.group2 = nn.Sequential(
            nn.Linear(context_dim, reduced_dim),
            nn.ReLU(),
            nn.Linear(reduced_dim, hidden_dim),
        )
        nn.init.zeros_(self.group2[-1].weight)
        nn.init.zeros_(self.group2[-1].bias)

    def forward(self, value: torch.Tensor) -> tuple[float, torch.Tensor]:
        return 0.0, self.group2(value)


class CrossAttentionAdapterProcessor(nn.Module):
    """Wan2.1 camera AdaLN policy installed into canonical cross-attention."""

    def __init__(self, context_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.pose_inject_method = "adaln"
        self.k_proj = GroupLinearDualK(context_dim, hidden_dim)
        self.v_proj = GroupLinearDualV(context_dim, hidden_dim)

    def forward(
        self,
        attention: nn.Module,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        *,
        plucker_fea: torch.Tensor | None = None,
        plucker_context_lens: torch.Tensor | None = None,
        pose_scale: float = 1.0,
        **kwargs: Any,
    ) -> torch.Tensor:
        del plucker_context_lens, kwargs
        if attention.has_image_input:
            image_context, text_context = context[:, :257], context[:, 257:]
        else:
            image_context, text_context = None, context
        query = attention.norm_q(attention.q(hidden_states))
        key = attention.norm_k(attention.k(text_context))
        value = attention.v(text_context)
        output = attention.attn(query, key, value)
        if image_context is not None:
            image_key = attention.norm_k_img(attention.k_img(image_context))
            image_value = attention.v_img(image_context)
            output = output + attention.attn(query, image_key, image_value)

        if plucker_fea is not None and bool(torch.count_nonzero(plucker_fea)):
            projected_camera, projected_hidden = self.k_proj(plucker_fea, output)
            scale, shift = self.v_proj(projected_camera + projected_hidden)
            output = output * (1.0 + scale * pose_scale) + shift * pose_scale
        return attention.o(output)


class CameraPoseEncoder(nn.Module):
    """Inference-only spatiotemporal encoder for per-pixel Plücker features."""

    def __init__(
        self,
        *,
        context_dim: int = 2048,
        dim: int = 5120,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        in_channels: int = 6,
        downscale_coefficient: int = 8,
    ) -> None:
        super().__init__()
        start_channels = in_channels * downscale_coefficient**2
        self.pose_inject_method = "adaln"
        self.unshuffle = nn.PixelUnshuffle(downscale_coefficient)
        self.controlnet_encode_first = nn.Sequential(
            nn.Conv2d(start_channels, start_channels, kernel_size=1),
            nn.GroupNorm(2, start_channels),
            nn.Conv2d(start_channels, start_channels, kernel_size=1),
            nn.GroupNorm(2, start_channels),
            nn.ReLU(),
        )
        self.controlnet_encode_second = nn.Sequential(
            nn.Conv2d(start_channels, start_channels * 2, kernel_size=1),
            nn.GroupNorm(2, start_channels * 2),
            nn.ReLU(),
        )
        self.patch_embedding = nn.Conv3d(
            start_channels * 2,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, context_dim),
            nn.LayerNorm(context_dim),
        )

    def _compress_feature_time(
        self,
        value: torch.Tensor,
        *,
        batch: int,
        frames: int,
    ) -> torch.Tensor:
        _, _, height, width = value.shape
        value = rearrange(value, "(b f) c h w -> (b h w) c f", b=batch, f=frames)
        if value.shape[-1] % 2:
            first, rest = value[..., :1], value[..., 1:]
            if rest.shape[-1]:
                rest = F.avg_pool1d(rest, kernel_size=2, stride=2)
            value = torch.cat((first, rest), dim=-1)
        else:
            value = F.avg_pool1d(value, kernel_size=2, stride=2)
        return rearrange(value, "(b h w) c f -> (b f) c h w", b=batch, h=height, w=width)

    def forward(self, plucker: torch.Tensor) -> torch.Tensor:
        batch, frames = plucker.shape[:2]
        value = rearrange(plucker, "b f h w c -> (b f) c h w")
        value = self.controlnet_encode_first(self.unshuffle(value))
        value = self._compress_feature_time(value, batch=batch, frames=frames)
        frames = value.shape[0] // batch
        value = self.controlnet_encode_second(value)
        value = self._compress_feature_time(value, batch=batch, frames=frames)
        value = rearrange(value, "(b f) c h w -> b c f h w", b=batch)
        value = self.patch_embedding(value)
        value = rearrange(value, "b c f h w -> b (f h w) c").contiguous()
        return self.fc(value)


class FantasyWorldCameraCondition(nn.Module):
    """Install released Wan2.1 camera processors and own their pose encoder."""

    _CHANNELS_BY_INFORMATION = {"all": 12, "rgb_conf": 4, "plucker": 6}

    def __init__(
        self,
        wan_dit: nn.Module,
        *,
        pose_in_dim: int = 768,
        plucker_fea_dim: int = 2048,
        pose_inject_method: str = "adaln",
        use_info: str = "plucker",
        processor_layers: int = 25,
    ) -> None:
        super().__init__()
        if pose_inject_method != "adaln":
            raise ValueError("the released FantasyWorld inference checkpoint uses adaln camera injection")
        try:
            in_channels = self._CHANNELS_BY_INFORMATION[use_info]
        except KeyError as error:
            raise ValueError(f"unsupported FantasyWorld camera information mode: {use_info!r}") from error
        self.pose_in_dim = pose_in_dim
        self.plucker_fea_dim = plucker_fea_dim
        self.pose_inject_method = pose_inject_method
        self.use_info = use_info
        self.proj_model = nn.Identity()
        for block in list(wan_dit.blocks)[:processor_layers]:
            block.cross_attn.set_processor(
                CrossAttentionAdapterProcessor(
                    context_dim=plucker_fea_dim,
                    hidden_dim=wan_dit.dim,
                )
            )
        self.pose_encoder = CameraPoseEncoder(
            context_dim=plucker_fea_dim,
            dim=wan_dit.dim,
            patch_size=tuple(wan_dit.patch_size),
            in_channels=in_channels,
        )

    def get_pose_fea(self, plucker: torch.Tensor | None) -> torch.Tensor | None:
        return None if plucker is None else self.pose_encoder(plucker)


class FantasyWorldFusionModel(nn.Module):
    """Released FantasyWorld composite with native Wan and VGGT roles."""

    def __init__(
        self,
        *,
        pipe: nn.Module,
        vggt: nn.Module,
        camera_condition: FantasyWorldCameraCondition | None = None,
        start_index: int = 16,
        fusion_blocks: Sequence[int] | None = None,
        fusion_dim: int = 1152,
        fusion_heads: int = 12,
    ) -> None:
        super().__init__()
        self.pipe = pipe
        self.vggt = vggt
        self.camera_control = camera_condition is not None
        if camera_condition is not None:
            self.camera_condition = camera_condition
        self.start_index = int(start_index)
        remaining_blocks = len(self.pipe.dit.blocks) - self.start_index
        self.cross_attention_list = list(
            range(remaining_blocks) if fusion_blocks is None else fusion_blocks
        )
        if any(index < 0 or index >= remaining_blocks for index in self.cross_attention_list):
            raise ValueError("FantasyWorld fusion block index is outside the remaining Wan stack")
        if max(self.cross_attention_list, default=-1) >= len(self.vggt.aggregator.global_blocks):
            raise ValueError("FantasyWorld fusion requires a matching VGGT global block")
        self._fusion_positions = {
            block_index: position
            for position, block_index in enumerate(self.cross_attention_list)
        }
        self.bicross_dim = fusion_dim
        self.bicross_num_heads = fusion_heads
        self.freqs_bicross = complex_rotary_frequencies_3d(fusion_dim // fusion_heads)

        irg_blocks = nn.ModuleList()
        for block_index in self.cross_attention_list:
            wan_index = self.start_index + block_index
            wan_block = self.pipe.dit.blocks[wan_index]
            aggregate_block = self.vggt.aggregator.global_blocks[block_index]
            self.pipe.dit.blocks[wan_index] = nn.Identity()
            self.vggt.aggregator.global_blocks[block_index] = nn.Identity()
            irg_blocks.append(
                IRGBlock(
                    wan_block,
                    aggregate_block,
                    modality_one_dim=self.pipe.dit.dim,
                    modality_two_dim=self.vggt.embed_dim,
                    hidden_size=fusion_dim,
                    num_heads=fusion_heads,
                )
            )
        # Capitalization is retained because it is part of the released schema.
        self.IRGBlock = irg_blocks

    @staticmethod
    def _grid_frequencies(
        frequencies_3d: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        grid_size: tuple[int, int, int],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        frames, height, width = grid_size
        temporal, vertical, horizontal = frequencies_3d
        return torch.cat(
            (
                temporal[:frames].view(frames, 1, 1, -1).expand(frames, height, width, -1),
                vertical[:height].view(1, height, 1, -1).expand(frames, height, width, -1),
                horizontal[:width].view(1, 1, width, -1).expand(frames, height, width, -1),
            ),
            dim=-1,
        ).reshape(frames * height * width, 1, -1).to(device)

    def joint_forward(
        self,
        x: torch.Tensor,
        *,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        camera_token: torch.Tensor | None = None,
        plucker_fea: torch.Tensor | None = None,
        plucker_context_lens: torch.Tensor | None = None,
        control_camera_latents_input: torch.Tensor | None = None,
        uncond: bool = False,
        return_prediction: bool = False,
        use_gradient_checkpointing: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        del use_gradient_checkpointing
        dit = self.pipe.dit
        if timestep.ndim != 1:
            raise ValueError("FantasyWorld timestep must contain one value per sample")
        time_embedding = dit.time_embedding(
            sinusoidal_embedding_1d(dit.freq_dim, timestep).to(dtype=x.dtype)
        )
        time_modulation = dit.time_projection(time_embedding).unflatten(1, (6, dit.dim))
        condition = dit.prepare_condition_context(
            context,
            clip_feature=clip_feature,
        )
        if y is not None and dit.require_vae_embedding:
            x = torch.cat((x, y), dim=1)
        elif dit.has_image_input and dit.require_vae_embedding:
            raise ValueError("FantasyWorld Wan2.1 requires VAE image-condition latents")
        x, grid_size = dit.patchify(
            x,
            control_camera_latents_input=control_camera_latents_input,
        )
        frames, height, width = grid_size
        wan_frequencies = dit.rotary_frequencies(grid_size, device=x.device)
        fusion_frequencies_dit = self._grid_frequencies(
            self.freqs_bicross,
            grid_size,
            device=x.device,
        )
        fusion_frequencies_agg = build_frequencies_with_frame_tokens(
            self.freqs_bicross,
            frames,
            height,
            width,
            extra_tokens_per_frame=5,
            device=x.device,
        )
        wan_kwargs = dict(kwargs)
        wan_kwargs.update(
            plucker_fea=plucker_fea,
            plucker_context_lens=plucker_context_lens,
        )

        for block in list(dit.blocks)[: self.start_index]:
            x = block(x, condition, time_modulation, wan_frequencies, **wan_kwargs)

        vggt_input = rearrange(
            x,
            "b (f h w) c -> b c f h w",
            f=frames,
            h=height,
            w=width,
        )
        patch_token, camera_token, e0 = self.vggt._process_wan_input(
            patch_token=vggt_input,
            camera_token=camera_token,
            t=timestep,
        )
        tokens, pos = self.vggt.aggregator._process_aggregator_input(
            patch_token,
            camera_token,
        )
        batch, sequence, _, _, aggregate_dim = patch_token.shape
        _, patches_per_frame, _ = tokens.shape
        frame_index = 0
        global_index = 0
        prediction_features: list[torch.Tensor] = []

        for block_index in range(len(dit.blocks) - self.start_index):
            tokens, frame_index, frame_features = self.vggt.aggregator._process_frame_attention(
                tokens,
                batch,
                sequence,
                patches_per_frame,
                aggregate_dim,
                frame_index,
                pos=pos,
                e0=e0,
            )
            fusion_position = self._fusion_positions.get(block_index)
            if fusion_position is not None:
                x, tokens, global_features = self.IRGBlock[fusion_position](
                    x,
                    tokens,
                    context=condition,
                    t_mod=time_modulation,
                    frequencies_dit=wan_frequencies,
                    fusion_frequencies_dit=fusion_frequencies_dit,
                    fusion_frequencies_agg=fusion_frequencies_agg,
                    pos=pos,
                    e0=e0,
                    unconditioned_geometry=uncond,
                    **wan_kwargs,
                )
                global_index += 1
            else:
                x = dit.blocks[self.start_index + block_index](
                    x,
                    condition,
                    time_modulation,
                    wan_frequencies,
                    **wan_kwargs,
                )
                tokens, global_index, global_features = self.vggt.aggregator._process_global_attention(
                    tokens,
                    batch,
                    sequence,
                    patches_per_frame,
                    aggregate_dim,
                    global_index,
                    pos=pos,
                    e0=e0,
                )
            prediction_features.extend(
                torch.cat((frame_value, global_value), dim=-1)
                for frame_value, global_value in zip(frame_features, global_features)
            )

        output = dit.unpatchify(dit.head(x, time_embedding), grid_size)
        if not return_prediction:
            return output, None
        prediction = self.vggt._head_predction(
            patch_token,
            self.vggt.aggregator.patch_start_idx,
            prediction_features,
        )
        return output, prediction


__all__ = [
    "BiMultiHeadAttention",
    "CameraPoseEncoder",
    "CrossAttentionAdapterProcessor",
    "CrossModalityBiAttentionBlock",
    "FantasyWorldCameraCondition",
    "FantasyWorldFusionModel",
    "IRGBlock",
    "build_frequencies_with_frame_tokens",
]
