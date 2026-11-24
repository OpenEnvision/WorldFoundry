import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    import flash_attn_interface  # noqa: F401 - Import checks optional dependency availability.

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn  # noqa: F401 - Import checks optional dependency availability.

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn  # noqa: F401 - Import checks optional dependency availability.

    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
from worldfoundry.core.attention import (
    complex_rotary_frequencies_3d as precompute_freqs_cis_3d,
)
from worldfoundry.core.nn import scale_shift as modulate
from worldfoundry.core.nn import sinusoidal_embedding_1d

from .wan_video_dit import SelfAttention

# isort: split
from worldfoundry.base_models.diffusion_model.models.networks.wan import (
    MLP,
    CrossAttention,
    Head,
    WanModelStateDictConverter,
)


class ModalityProcessor(nn.Module):
    """Project one modality's conditioning input to the shared MoE input dimension."""

    def __init__(self, modality_name: str, input_dim: int, unified_dim: int = 30):
        super().__init__()
        self.modality_name = modality_name
        self.input_dim = input_dim
        self.unified_dim = unified_dim

        self.projector = nn.Sequential(nn.Linear(input_dim, unified_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim] or [batch_size, input_dim]
        Returns:
            projected: [batch_size, seq_len, unified_dim]
        """
        original_dtype = x.dtype

        if x.dim() == 2:  # [batch, input_dim]
            x = x.unsqueeze(1)  # [batch, 1, input_dim]

        # Run the projection in the projector's dtype and return the input dtype.
        x = x.to(self.projector[0].weight.dtype)

        output = self.projector(x)

        output = output.to(original_dtype)

        return output


class MultiModalMoE(nn.Module):
    """Multi-modal MoE experts; routing decisions are passed in by the caller."""

    def __init__(
        self,
        unified_dim: int = 30,
        hidden_dim: int = 60,
        output_dim: int | None = None,
        num_experts: int = 4,
        top_k: int = 2,
    ):
        super().__init__()
        self.unified_dim = unified_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.output_dim = output_dim or unified_dim

        # Expert assigned to each modality; unknown modalities use expert 0.
        self.modality_to_expert = {
            "sekai": 0,
            "nuscenes": 1,
            "openx": 2,
            "unknown": 0,
        }

        # Per-layer experts mapping unified_dim to output_dim.
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(unified_dim, self.output_dim)) for _ in range(num_experts)]
        )

    def forward(
        self, x: torch.Tensor, expert_weights: torch.Tensor, top_k_indices: torch.Tensor, modality_type: str = "unknown"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, unified_dim]
            expert_weights: [batch_size, seq_len, top_k] - weights from the global router
            top_k_indices: [batch_size, seq_len, top_k] - expert indices from the global router
            modality_type: modality name, used for the target expert and statistics
        Returns:
            output: [batch_size, seq_len, output_dim]
            expert_stats: expert selection statistics
        """
        batch_size, seq_len, input_dim = x.shape
        assert input_dim == self.unified_dim, f"Expected input dim {self.unified_dim}, got {input_dim}"

        original_dtype = x.dtype
        x = x.to(self.experts[0][0].weight.dtype)

        target_expert_id = self.modality_to_expert.get(modality_type, 0)

        expert_stats = self.collect_expert_statistics(expert_weights, top_k_indices, modality_type, target_expert_id)

        # Expert processing with this layer's experts
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(x)  # [batch, seq, output_dim]
            expert_outputs.append(expert_output)

        expert_outputs = torch.stack(expert_outputs, dim=-2)  # [batch, seq, num_experts, output_dim]

        # Weighted combination using provided weights and indices
        output = torch.zeros(batch_size, seq_len, self.output_dim, device=x.device, dtype=x.dtype)

        for k in range(self.top_k):
            expert_idx = top_k_indices[:, :, k]  # [batch, seq]
            weight = expert_weights[:, :, k : k + 1]  # [batch, seq, 1]

            expert_output = torch.gather(
                expert_outputs,
                dim=2,
                index=expert_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, expert_outputs.shape[-1]),
            ).squeeze(2)  # [batch, seq, output_dim]

            output += weight * expert_output

        output = output.to(original_dtype)

        return output, expert_stats

    def collect_expert_statistics(self, expert_weights, top_k_indices, modality_type, target_expert_id):
        """Summarize expert selection frequencies and weights for logging."""
        with torch.no_grad():
            # Selection frequency of each expert
            expert_selection_count = torch.zeros(self.num_experts, device=expert_weights.device)
            for expert_id in range(self.num_experts):
                expert_selection_count[expert_id] = (top_k_indices == expert_id).float().sum()

            total_selections = expert_selection_count.sum()
            expert_selection_ratio = expert_selection_count / (total_selections + 1e-8)

            # Mean routing weight of each expert
            avg_expert_weights = torch.zeros(self.num_experts, device=expert_weights.device)
            for expert_id in range(self.num_experts):
                mask = top_k_indices == expert_id
                if mask.sum() > 0:
                    avg_expert_weights[expert_id] = expert_weights[mask].mean()

            avg_top_k_weights = expert_weights.mean(dim=(0, 1))

            target_expert_usage = expert_selection_ratio[target_expert_id].item()

            return {
                "modality_type": modality_type,
                "target_expert_id": target_expert_id,
                "target_expert_usage": target_expert_usage,
                "expert_selection_ratio": expert_selection_ratio.float().cpu().numpy(),
                "avg_expert_weights": avg_expert_weights.float().cpu().numpy(),
                "avg_top_k_weights": avg_top_k_weights.float().cpu().numpy(),
                "num_experts": self.num_experts,
                "top_k": self.top_k,
            }


class DiTBlockWithMoE(nn.Module):
    """DiT block that adds the MoE output of the modality inputs to the modulated input."""

    def __init__(
        self,
        has_image_input: bool,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
        use_moe: bool = True,
        moe_config: dict | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.use_moe = use_moe

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # MoE experts are created only when MoE is enabled and configured.
        if self.use_moe and moe_config:
            unified_dim = moe_config.get("unified_dim", 30)
            self.moe = MultiModalMoE(
                unified_dim=unified_dim,
                output_dim=dim,  # Matches the block dim for the residual connection
                num_experts=moe_config.get("num_experts", 4),
                top_k=moe_config.get("top_k", 2),
            )

    def forward(
        self,
        x,
        context,
        cam_emb,
        t_mod,
        freqs,
        modality_inputs: dict | None = None,
        router_weights: torch.Tensor | None = None,
        router_indices: torch.Tensor | None = None,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)

        # MoE path with the global routing decisions
        if self.use_moe and modality_inputs and hasattr(self, "moe") and router_weights is not None:
            # Sum the processed inputs of all modalities.
            combined_modality_input = None
            active_modality = "unknown"
            for modality_type, processed_input in modality_inputs.items():
                active_modality = modality_type
                if combined_modality_input is None:
                    combined_modality_input = processed_input
                else:
                    combined_modality_input = combined_modality_input + processed_input

            if combined_modality_input is not None:
                moe_output, expert_stats = self.moe(
                    combined_modality_input, router_weights, router_indices, active_modality
                )
                input_x = input_x + moe_output

                # Buffer expert statistics for print_overall_expert_statistics.
                if not hasattr(self, "expert_stats_buffer"):
                    self.expert_stats_buffer = []

                self.expert_stats_buffer.append(expert_stats)
        elif cam_emb is not None and hasattr(self, "cam_encoder"):
            # Fallback to the plain camera encoder
            cam_emb = cam_emb.to(self.cam_encoder.weight.dtype)
            cam_emb = self.cam_encoder(cam_emb)
            input_x = input_x + cam_emb

        input_x = input_x.to(self.projector.weight.dtype)

        # Ensure self.self_attn output dtype matches self.projector.weight dtype
        attn_output = self.self_attn(input_x, freqs)
        attn_output = attn_output.to(self.projector.weight.dtype)

        x = x + gate_msa * self.projector(attn_output)
        x = x.to(self.norm3.weight.dtype)
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.ffn(input_x)
        return x


class WanModelMoe(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        # MoE configuration
        use_moe: bool = True,
        moe_config: dict | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.use_moe = use_moe
        self.moe_config = moe_config or {}

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # # Global router shared by all blocks
        # if use_moe and moe_config:
        #     unified_dim = moe_config.get("unified_dim", 30)
        #     num_experts = moe_config.get("num_experts", 4)
        #     self.top_k = moe_config.get("top_k", 2)

        # Expert assigned to each modality; unknown modalities use expert 0.
        self.modality_to_expert = {
            "sekai": 0,
            "nuscenes": 1,
            "openx": 2,
            "unknown": 0,
        }
        self.top_k = 1

        #     self.global_router = nn.Linear(unified_dim, num_experts)
        #     print(f"Created the global router: input_dim={unified_dim}, num_experts={num_experts}")
        # else:
        #     self.global_router = None
        #     self.modality_to_expert = {}

        self.blocks = nn.ModuleList(
            [
                DiTBlockWithMoE(has_image_input, dim, num_heads, ffn_dim, eps, use_moe, moe_config)
                for _ in range(num_layers)
            ]
        )

        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim)  # clip_feature_dim = 1280

    def compute_router_decisions(self, combined_modality_input: torch.Tensor, modality_type: str):
        """
        Route every token to the modality's fixed expert from modality_to_expert with weight 1.
        """
        batch_size, seq_len, _ = combined_modality_input.shape
        _num_experts = len(self.modality_to_expert)
        top_k = self.top_k if hasattr(self, "top_k") else 1

        target_expert_id = self.modality_to_expert.get(modality_type, 0)

        router_indices = torch.full(
            (batch_size, seq_len, top_k), target_expert_id, dtype=torch.long, device=combined_modality_input.device
        )
        router_weights = torch.ones(
            (batch_size, seq_len, top_k), dtype=combined_modality_input.dtype, device=combined_modality_input.device
        )

        # Fixed routing has no specialization loss.
        specialization_loss = torch.tensor(0.0, device=combined_modality_input.device)

        return router_weights, router_indices, specialization_loss

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

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

    def create_clean_x_embedder(self):
        """Create the FramePack-style embedder for clean latents at 1x, 2x and 4x scales."""

        class CleanXEmbedder(nn.Module):
            def __init__(self, inner_dim):
                super().__init__()
                # Follows hunyuan_video_packed.py.
                self.proj = nn.Conv3d(16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                self.proj_2x = nn.Conv3d(16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
                self.proj_4x = nn.Conv3d(16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))

            def forward(self, x, scale="1x"):
                if scale == "1x":
                    return self.proj(x)
                elif scale == "2x":
                    return self.proj_2x(x)
                elif scale == "4x":
                    return self.proj_4x(x)
                else:
                    raise ValueError(f"Unsupported scale: {scale}")

        return CleanXEmbedder(self.dim)

    def rope(self, frame_indices, height, width, device):
        """Build 3D RoPE frequencies for the given frame indices, following HunyuanVideo."""
        batch_size = frame_indices.shape[0]
        seq_len = frame_indices.shape[1]

        # Temporal frequencies at frame_indices
        f_freqs = self.freqs[0][frame_indices.to("cpu")]  # [batch, seq_len, freq_dim]

        h_positions = torch.arange(height, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        w_positions = torch.arange(width, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)

        h_freqs = self.freqs[1][h_positions.to("cpu")]  # [batch, seq_len, height, h_freq_dim]
        w_freqs = self.freqs[2][w_positions.to("cpu")]  # [batch, seq_len, width, w_freq_dim]

        # Broadcast over the full spatial grid.
        f_freqs_expanded = f_freqs.unsqueeze(2).unsqueeze(3).expand(-1, -1, height, width, -1)
        h_freqs_expanded = h_freqs.unsqueeze(3).expand(-1, -1, -1, width, -1)
        w_freqs_expanded = w_freqs.unsqueeze(2).expand(-1, -1, height, -1, -1)

        rope_freqs = torch.cat([f_freqs_expanded, h_freqs_expanded, w_freqs_expanded], dim=-1)

        return rope_freqs  # [batch, seq_len, height, width, total_freq_dim]

    def pad_for_3d_conv(self, x, kernel_size):
        """Replicate-pad the end of T, H and W to multiples of kernel_size, following HunyuanVideo."""
        if len(x.shape) == 5:  # [B, C, T, H, W]
            b, c, t, h, w = x.shape
            pt, ph, pw = kernel_size
            pad_t = (pt - (t % pt)) % pt
            pad_h = (ph - (h % ph)) % ph
            pad_w = (pw - (w % pw)) % pw
            return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")
        elif len(x.shape) == 6:  # [B, T, H, W, C] (RoPE frequencies)
            b, t, h, w, c = x.shape
            pt, ph, pw = kernel_size
            pad_t = (pt - (t % pt)) % pt
            pad_h = (ph - (h % ph)) % ph
            pad_w = (pw - (w % pw)) % pw
            return torch.nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")
        else:
            raise ValueError(f"Unsupported tensor shape: {x.shape}")

    def center_down_sample_3d(self, x, scale_factor):
        """Strided 3D downsampling, following HunyuanVideo's center_down_sample_3d."""
        if len(x.shape) == 6:  # [B, T, H, W, C] (RoPE frequencies)
            st, sh, sw = scale_factor
            return x[:, ::st, ::sh, ::sw, :]
        elif len(x.shape) == 5:  # [B, C, T, H, W]
            st, sh, sw = scale_factor
            return x[:, :, ::st, ::sh, ::sw]
        else:
            raise ValueError(f"Unsupported tensor shape: {x.shape}")

    def process_modality_inputs(self, modality_inputs):
        """Project each modality input with its processor."""
        if not modality_inputs or not self.use_moe:
            return None

        processed_modality_inputs = {}

        for modality_type, input_data in modality_inputs.items():
            if modality_type == "sekai" and hasattr(self, "sekai_processor"):
                processed = self.sekai_processor(input_data)
                processed_modality_inputs[modality_type] = processed
            elif modality_type == "nuscenes" and hasattr(self, "nuscenes_processor"):
                processed = self.nuscenes_processor(input_data)
                processed_modality_inputs[modality_type] = processed
            elif modality_type == "openx" and hasattr(self, "openx_processor"):
                processed = self.openx_processor(input_data)
                processed_modality_inputs[modality_type] = processed
            else:
                print(f"Unknown modality type: {modality_type}")
                continue

        return processed_modality_inputs, processed

    def _frame_rope_freqs(self, t_idx, h, w):
        """RoPE frequencies of one latent frame at temporal position ``t_idx``, shaped [h*w, total_freq_dim]."""
        f_freq = self.freqs[0][t_idx : t_idx + 1]  # [1, freq_dim]
        h_freq = self.freqs[1][:h]  # [h, freq_dim]
        w_freq = self.freqs[2][:w]  # [w, freq_dim]
        return torch.cat(
            [
                f_freq.view(1, 1, 1, -1).expand(1, h, w, -1),
                h_freq.view(1, h, 1, -1).expand(1, h, w, -1),
                w_freq.view(1, 1, w, -1).expand(1, h, w, -1),
            ],
            dim=-1,
        ).reshape(h * w, -1)

    def _indexed_rope_freqs(self, frame_indices, batch_size, h, w):
        """RoPE frequencies for per-sample temporal positions ``frame_indices`` [B, f], shaped [B, f*h*w, total_freq_dim]."""
        return torch.stack(
            [
                torch.cat([self._frame_rope_freqs(t_idx, h, w) for t_idx in frame_indices[b]], dim=0)
                for b in range(batch_size)
            ],
            dim=0,
        )

    def _prepend_downsampled_clean_latents(
        self,
        hidden_states,
        rope_freqs,
        combined_modality_embeddings,
        clean_latents_scaled,
        clean_latent_scaled_indices,
        scale,
        kernel_size,
        cam_emb,
        start_indice,
    ):
        """Prepend the 2x or 4x clean latents with their RoPE frequencies and camera embeddings.

        Valid indices are non-negative; -1 marks padding. Inputs are returned unchanged when there are no
        valid indices.
        """
        if (
            clean_latents_scaled is None
            or clean_latent_scaled_indices is None
            or clean_latent_scaled_indices.numel() == 0
        ):
            return hidden_states, rope_freqs, combined_modality_embeddings
        valid_indices = clean_latent_scaled_indices[clean_latent_scaled_indices >= 0]
        if len(valid_indices) == 0:
            return hidden_states, rope_freqs, combined_modality_embeddings

        B = hidden_states.shape[0]
        clean_latents_scaled = clean_latents_scaled.to(hidden_states)
        clean_latents_scaled = self.pad_for_3d_conv(clean_latents_scaled, kernel_size)
        clean_hidden_states = self.clean_x_embedder(clean_latents_scaled, scale=scale)

        _, _, clean_f, clean_h, clean_w = clean_hidden_states.shape
        clean_hidden_states = rearrange(clean_hidden_states, "b c f h w -> b (f h w) c")

        # RoPE frequencies for the downsampled latents; frames past the valid indices reuse the last one.
        frame_positions = [valid_indices[min(frame_idx, len(valid_indices) - 1)] for frame_idx in range(clean_f)]
        clean_rope_freqs = torch.stack(
            [
                torch.cat([self._frame_rope_freqs(t_idx, clean_h, clean_w) for t_idx in frame_positions], dim=0)
                for _ in range(B)
            ],
            dim=0,
        )

        if cam_emb is not None:
            # Camera embeddings, zero for frames without a valid index
            clean_camera = torch.zeros(B, clean_f, cam_emb.shape[-1], dtype=cam_emb.dtype, device=cam_emb.device)
            for frame_idx in range(min(clean_f, len(valid_indices))):
                cam_idx = valid_indices[frame_idx].item() - start_indice
                if 0 <= cam_idx < cam_emb.shape[1]:
                    clean_camera[:, frame_idx, :] = cam_emb[:, cam_idx, :]

            clean_camera_spatial = clean_camera.unsqueeze(2).unsqueeze(3).repeat(1, 1, clean_h, clean_w, 1)
            clean_camera_spatial = rearrange(clean_camera_spatial, "b f h w d -> b (f h w) d")
            combined_modality_embeddings = torch.cat([clean_camera_spatial, combined_modality_embeddings], dim=1)

        hidden_states = torch.cat([clean_hidden_states, hidden_states], dim=1)
        rope_freqs = torch.cat([clean_rope_freqs, rope_freqs], dim=1)
        return hidden_states, rope_freqs, combined_modality_embeddings

    def process_input_hidden_states(
        self,
        latents,
        latent_indices=None,
        clean_latents=None,
        clean_latent_indices=None,
        clean_latents_2x=None,
        clean_latent_2x_indices=None,
        clean_latents_4x=None,
        clean_latent_4x_indices=None,
        cam_emb=None,
        modality_inputs: dict | None = None,
    ):
        """Embed FramePack-style multi-scale latents with their RoPE frequencies and camera embeddings, following wan_video_dit_recam_future."""

        # Target latents
        hidden_states, grid_size = self.patchify(latents)
        B, T_patches, C = hidden_states.shape
        f, h, w = grid_size

        # RoPE frequencies use the temporal positions in latent_indices.
        if latent_indices is None:
            latent_indices = torch.arange(0, f, device=hidden_states.device).unsqueeze(0).expand(B, -1)

        rope_freqs = self._indexed_rope_freqs(latent_indices, B, h, w)  # [B, f*h*w, total_freq_dim]

        # Camera embeddings of the target latents (1x scale, h*w tokens per frame)
        start_indice = clean_latent_indices[0][0].item() if clean_latent_indices is not None else 0
        combined_modality_embeddings = None

        if cam_emb is not None:
            # Target camera frames selected by latent_indices
            target_start = latent_indices[0].min().item() - start_indice
            target_end = latent_indices[0].max().item() + 1 - start_indice
            target_camera = cam_emb[:, target_start:target_end, :]  # [B, target_frames, cam_dim]

            target_camera_spatial = target_camera.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
            target_camera_spatial = rearrange(target_camera_spatial, "b f h w d -> b (f h w) d")
            combined_modality_embeddings = target_camera_spatial

        # clean_latents (1x scale)
        if clean_latents is not None and clean_latent_indices is not None:
            clean_latents = clean_latents.to(hidden_states)
            clean_hidden_states = self.clean_x_embedder(clean_latents, scale="1x")
            clean_hidden_states = rearrange(clean_hidden_states, "b c f h w -> b (f h w) c")

            clean_rope_freqs = self._indexed_rope_freqs(clean_latent_indices, B, h, w)

            # Camera embeddings of the clean latents
            if cam_emb is not None:
                clean_start = clean_latent_indices[0].min().item() - start_indice
                clean_end = clean_latent_indices[0].max().item() + 1 - start_indice

                if clean_start == clean_end:
                    clean_camera = cam_emb[:, clean_start : clean_start + 1, :]  # [B, 1, cam_dim]
                else:
                    clean_camera = cam_emb[:, [clean_start, clean_end], :]  # [B, 2, cam_dim]

                clean_camera_spatial = clean_camera.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
                clean_camera_spatial = rearrange(clean_camera_spatial, "b f h w d -> b (f h w) d")
                combined_modality_embeddings = torch.cat([clean_camera_spatial, combined_modality_embeddings], dim=1)

            # Prepend the clean latents and their frequencies.
            hidden_states = torch.cat([clean_hidden_states, hidden_states], dim=1)
            rope_freqs = torch.cat([clean_rope_freqs, rope_freqs], dim=1)

        # clean_latents_2x (2x scale), then clean_latents_4x (4x scale)
        for scale, kernel_size, scaled_latents, scaled_indices in (
            ("2x", (2, 4, 4), clean_latents_2x, clean_latent_2x_indices),
            ("4x", (4, 8, 8), clean_latents_4x, clean_latent_4x_indices),
        ):
            hidden_states, rope_freqs, combined_modality_embeddings = self._prepend_downsampled_clean_latents(
                hidden_states,
                rope_freqs,
                combined_modality_embeddings,
                scaled_latents,
                scaled_indices,
                scale,
                kernel_size,
                cam_emb,
                start_indice,
            )

        rope_freqs = rope_freqs.unsqueeze(2).to(device=hidden_states.device)

        # Every modality receives the combined camera embeddings.
        processed_modality_inputs = None
        if modality_inputs and self.use_moe:
            processed_modality_inputs = {}
            for modality_type in modality_inputs.keys():  # noqa: SIM118 - Keeps the keys view for mappings with custom membership.
                processed_modality_inputs[modality_type] = combined_modality_embeddings

        return hidden_states, rope_freqs, grid_size, combined_modality_embeddings, processed_modality_inputs

    def forward(
        self,
        latents,
        timestep,
        cam_emb,
        # FramePack inputs
        latent_indices=None,
        clean_latents=None,
        clean_latent_indices=None,
        clean_latents_2x=None,
        clean_latent_2x_indices=None,
        clean_latents_4x=None,
        clean_latent_4x_indices=None,
        # MoE inputs
        modality_inputs: dict | None = None,
        **kwargs,
    ):
        modality_inputs, cam_emb = self.process_modality_inputs(modality_inputs)

        # Reset the expert statistics of the previous call.
        for block in self.blocks:
            if hasattr(block, "expert_stats_buffer"):
                block.expert_stats_buffer = []

        hidden_states, rope_freqs, grid_size, processed_cam_emb, processed_modality_inputs = (
            self.process_input_hidden_states(
                latents,
                latent_indices,
                clean_latents,
                clean_latent_indices,
                clean_latents_2x,
                clean_latent_2x_indices,
                clean_latents_4x,
                clean_latent_4x_indices,
                cam_emb,
                modality_inputs,
            )
        )

        # Token count of the target latents, used to slice the output
        batch_size, num_channels, num_frames, height, width = latents.shape
        p, p_t = self.patch_size[2], self.patch_size[0]  # [t, h, w]
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p
        original_context_length = post_patch_num_frames * post_patch_height * post_patch_width

        context = kwargs.get("context")
        if context is not None:
            context = self.text_embedding(context)
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        # t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        with torch.amp.autocast("cuda", enabled=False):
            # Force time projection (and parameters) to run in fp32 to bypass bf16 autocast
            t_fp32 = t.float()
            t_activated = self.time_projection[0](t_fp32)
            linear = self.time_projection[1]
            weight_fp32 = linear.weight.float()
            bias_fp32 = linear.bias.float() if linear.bias is not None else None
            t_proj = F.linear(t_activated, weight_fp32, bias_fp32)
            t_proj = t_proj.to(t.dtype)
        t_mod = t_proj.unflatten(1, (6, self.dim))

        assert rope_freqs.shape[1] == hidden_states.shape[1], (
            f"RoPE sequence length {rope_freqs.shape[1]} does not match the hidden_states sequence length {hidden_states.shape[1]}"
        )

        # Routing decisions are computed once for all blocks.
        router_weights, router_indices, total_specialization_loss = (
            None,
            None,
            torch.tensor(0.0, device=hidden_states.device),
        )
        active_modality = "unknown"

        if self.use_moe and processed_modality_inputs:
            combined_modality_input = None
            for modality_type, processed_input in processed_modality_inputs.items():
                active_modality = modality_type
                if combined_modality_input is None:
                    combined_modality_input = processed_input
                else:
                    combined_modality_input = combined_modality_input + processed_input

            # router_input = torch.cat([hidden_states, combined_modality_input], dim=-1)
            if combined_modality_input is not None:
                router_weights, router_indices, total_specialization_loss = self.compute_router_decisions(
                    combined_modality_input, active_modality
                )

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                context,
                processed_cam_emb,
                t_mod,
                rope_freqs,
                processed_modality_inputs,
                router_weights,
                router_indices,
            )

        # self.print_overall_expert_statistics()

        # Project only the target tokens.
        hidden_states = hidden_states[:, -original_context_length:, :]
        hidden_states = self.head(hidden_states, t)
        hidden_states = self.unpatchify(hidden_states, grid_size)

        return hidden_states, total_specialization_loss

    def print_overall_expert_statistics(self):
        """Print the expert statistics buffered by all blocks, grouped by modality."""
        all_expert_stats = []

        for i, block in enumerate(self.blocks):  # noqa: B007 - The loop variable is read outside the loop.
            if hasattr(block, "expert_stats_buffer") and len(block.expert_stats_buffer) > 0:
                all_expert_stats.extend(block.expert_stats_buffer)

        if not all_expert_stats:
            return

        modality_stats = {}
        for stats in all_expert_stats:
            modality = stats["modality_type"]
            if modality not in modality_stats:
                modality_stats[modality] = {
                    "selection_ratios": [],
                    "expert_weights": [],
                    "top_k_weights": [],
                    "target_expert_usages": [],
                    "target_expert_id": stats["target_expert_id"],
                    "count": 0,
                }

            modality_stats[modality]["selection_ratios"].append(stats["expert_selection_ratio"])
            modality_stats[modality]["expert_weights"].append(stats["avg_expert_weights"])
            modality_stats[modality]["top_k_weights"].append(stats["avg_top_k_weights"])
            modality_stats[modality]["target_expert_usages"].append(stats["target_expert_usage"])
            modality_stats[modality]["count"] += 1

        print("\n" + "=" * 60)
        print("Expert specialization statistics (global router + per-layer experts)")
        print("=" * 60)

        for modality, stats in modality_stats.items():
            if stats["count"] == 0:
                continue

            avg_selection_ratio = np.mean(stats["selection_ratios"], axis=0)
            avg_expert_weights = np.mean(stats["expert_weights"], axis=0)
            _avg_top_k_weights = np.mean(stats["top_k_weights"], axis=0)
            avg_target_expert_usage = np.mean(stats["target_expert_usages"])
            target_expert_id = stats["target_expert_id"]

            print(f"\n {modality.upper()} modality (Source {stats['count']} MoE blocks)")
            print(f"   expected expert: Expert-{target_expert_id}")
            print(f"   expected expert usage: {avg_target_expert_usage:.3f} ({avg_target_expert_usage * 100:.1f}%)")

            print("   Expert chosen weight (global Router decision):")
            for i, ratio in enumerate(avg_selection_ratio):
                status = "🔥" if i == target_expert_id else "  "
                print(f"    {status} Expert-{i}: {ratio:.3f} ({ratio * 100:.1f}%)")

            print("    Expert avg weight:")
            for i, weight in enumerate(avg_expert_weights):
                status = "🔥" if i == target_expert_id else "  "
                print(f"    {status} Expert-{i}: {weight:.3f}")

            if avg_target_expert_usage > 0.8:
                specialization_status = " high"
            elif avg_target_expert_usage > 0.5:
                specialization_status = " good"
            else:
                specialization_status = "  insufficient"

            print(f"   specialization: {specialization_status}")

            most_used_expert = np.argmax(avg_selection_ratio)
            most_used_ratio = avg_selection_ratio[most_used_expert]
            if most_used_expert == target_expert_id:
                print(f"   Actual most expert: Expert-{most_used_expert} ({most_used_ratio:.3f}) - OK!")
            else:
                print(f"   Actual most expert: Expert-{most_used_expert} ({most_used_ratio:.3f}) - No")

        print("=" * 60)

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()
