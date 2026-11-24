"""LTX dual-stream DiT block (self-attn, text CA, optional A↔V CA, FFN).

Tokens stay ``B T D`` throughout.  Video and audio each own AdaLN tables
and attention modules; joint checkpoints add audio-to-video and
video-to-audio attention.  A↔V uses pre-cross snapshots so direction
order does not leak into the other stream's keys.
"""

from dataclasses import dataclass, replace

import torch

from worldfoundry.base_models.diffusion_model.models.networks.ltx.adaln import adaln_embedding_coefficient
from worldfoundry.base_models.diffusion_model.models.networks.ltx.attention import (
    Attention,
)
from worldfoundry.base_models.diffusion_model.models.networks.ltx.feed_forward import FeedForward
from worldfoundry.base_models.diffusion_model.models.networks.ltx.rope import LTXRopeType
from worldfoundry.base_models.diffusion_model.models.networks.ltx.transformer_args import TransformerArgs
from worldfoundry.core.attention.backends.model_backends import AttentionCallable
from worldfoundry.core.nn import TransformerOpsConfig


@dataclass
class TransformerConfig:
    """Per-modality block widths: hidden dim, heads, head dim, text context dim."""

    dim: int
    heads: int
    d_head: int
    context_dim: int
    apply_gated_attention: bool = False
    cross_attention_adaln: bool = False
    ff_bias: bool = True


class BasicAVTransformerBlock(torch.nn.Module):
    """One LTX layer: per-stream MSA + text CA, optional bidirectional AV CA, FFN."""

    def __init__(
        self,
        video: TransformerConfig | None = None,
        audio: TransformerConfig | None = None,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        norm_eps: float = 1e-6,
        ops: TransformerOpsConfig | None = None,
        video_attention_class: type[Attention] = Attention,
    ):
        """Construct only the stems present in ``video`` / ``audio`` configs."""
        super().__init__()

        if ops is None:
            ops = TransformerOpsConfig()
        self.ada_zero_function = ops.ada_zero_function
        self.post_sa_function = ops.post_sa_function
        if video is not None:
            self.attn1 = video_attention_class(
                query_dim=video.dim,
                heads=video.heads,
                dim_head=video.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=ops.attention_ops,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.attn2 = Attention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=ops.attention_ops,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.ff = FeedForward(video.dim, dim_out=video.dim, bias=video.ff_bias)
            video_sst_size = adaln_embedding_coefficient(video.cross_attention_adaln)
            self.scale_shift_table = torch.nn.Parameter(torch.empty(video_sst_size, video.dim))

        if audio is not None:
            self.audio_attn1 = Attention(
                query_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=ops.attention_ops,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_attn2 = Attention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=ops.attention_ops,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim, bias=audio.ff_bias)
            audio_sst_size = adaln_embedding_coefficient(audio.cross_attention_adaln)
            self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(audio_sst_size, audio.dim))

        if audio is not None and video is not None:
            # Q: Video, K,V: Audio
            self.audio_to_video_attn = Attention(
                query_dim=video.dim,
                context_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=ops.attention_ops,
                apply_gated_attention=video.apply_gated_attention,
            )

            # Q: Audio, K,V: Video
            self.video_to_audio_attn = Attention(
                query_dim=audio.dim,
                context_dim=video.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=ops.attention_ops,
                apply_gated_attention=audio.apply_gated_attention,
            )

            self.scale_shift_table_a2v_ca_audio = torch.nn.Parameter(torch.empty(5, audio.dim))
            self.scale_shift_table_a2v_ca_video = torch.nn.Parameter(torch.empty(5, video.dim))

        self.cross_attention_adaln = (video is not None and video.cross_attention_adaln) or (
            audio is not None and audio.cross_attention_adaln
        )

        if self.cross_attention_adaln and video is not None:
            self.prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, video.dim))
        if self.cross_attention_adaln and audio is not None:
            self.audio_prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, audio.dim))

        self.norm_eps = norm_eps
        self._use_fused_residual_gate: bool = False

    def _all_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Compute every AdaLN modulation slice for this block in one fused addition."""
        num_ada_params = scale_shift_table.shape[0]
        return (
            scale_shift_table.unsqueeze(0).unsqueeze(0).to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)
        ).unbind(dim=2)

    def _add_gated_residual(
        self,
        residual: torch.Tensor,
        update: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        if self._use_fused_residual_gate:
            from worldfoundry.core.kernels.diffusion import residual_gate_add

            return residual_gate_add(residual, update, gate)
        return residual + update * gate

    def get_ada_values(
        self, scale_shift_table: torch.Tensor, batch_size: int, timestep: torch.Tensor, indices: slice
    ) -> tuple[torch.Tensor, ...]:
        """Slice AdaLN ``(shift, scale, gate, ...)`` for this residual branch."""
        num_ada_params = scale_shift_table.shape[0]

        ada_values = (
            scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        ).unbind(dim=2)
        return ada_values

    def get_av_ca_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        scale_shift_indices: slice,
        num_scale_shift_values: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """AV cross-attn AdaLN: scale/shift from one timestep, gate from the other modality's sigma."""
        scale_shift_ada_values = self.get_ada_values(
            scale_shift_table[:num_scale_shift_values, :], batch_size, scale_shift_timestep, scale_shift_indices
        )
        gate_ada_values = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :], batch_size, gate_timestep, slice(None, None)
        )

        scale, shift = (t.squeeze(2) for t in scale_shift_ada_values)
        (gate,) = (t.squeeze(2) for t in gate_ada_values)

        return scale, shift, gate

    def _apply_text_cross_attention(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attn: AttentionCallable,
        scale_shift_table: torch.Tensor,
        prompt_scale_shift_table: torch.Tensor | None,
        timestep: torch.Tensor,
        prompt_timestep: torch.Tensor | None,
        context_mask: torch.Tensor | None,
        cross_attention_adaln: bool = False,
        ca_ada_values: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Apply text cross-attention to normalized post-self-attention tokens."""
        if cross_attention_adaln:
            if ca_ada_values is not None:
                shift_q, scale_q, gate = ca_ada_values
            else:
                shift_q, scale_q, gate = self.get_ada_values(scale_shift_table, x.shape[0], timestep, slice(6, 9))
            return apply_cross_attention_adaln(
                x,
                context,
                attn,
                shift_q,
                scale_q,
                gate,
                prompt_scale_shift_table,
                prompt_timestep,
                context_mask,
            )
        return attn(x, context=context, mask=context_mask)

    def forward(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        """Update ``TransformerArgs.x`` for each enabled stream; empty tensors are skipped."""
        if video is None and audio is None:
            raise ValueError("At least one of video or audio must be provided")

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax.numel() > 0

        run_a2v = run_vx and (audio is not None and ax.numel() > 0)
        run_v2a = run_ax and (video is not None and vx.numel() > 0)

        v_ada: tuple[torch.Tensor, ...] | None = None
        a_ada: tuple[torch.Tensor, ...] | None = None

        if run_vx:
            assert video is not None and vx is not None
            v_ada = self._all_ada_values(self.scale_shift_table, vx.shape[0], video.timesteps)
            vshift_msa, vscale_msa, vgate_msa = v_ada[0], v_ada[1], v_ada[2]
            norm_vx = self.ada_zero_function(vx, self.norm_eps, vscale_msa, vshift_msa)
            del vshift_msa, vscale_msa

            vx_msa_out = self.attn1(
                norm_vx,
                pe=video.positional_embeddings,
                mask=video.self_attention_mask,
                perturbation_mask=video.self_attn_perturbation_mask,
                all_perturbed=video.self_attn_all_perturbed,
                camera_control=video.video_control,
                cache=video.attention_cache,
            )
            vx, vx_normed = self.post_sa_function(vx, vx_msa_out, None, self.norm_eps, vgate_msa)
            del vgate_msa, norm_vx, vx_msa_out
            vx = vx + self._apply_text_cross_attention(
                vx_normed,
                video.context,
                self.attn2,
                self.scale_shift_table,
                getattr(self, "prompt_scale_shift_table", None),
                video.timesteps,
                video.prompt_timestep,
                video.context_mask,
                cross_attention_adaln=self.cross_attention_adaln,
                ca_ada_values=(v_ada[6], v_ada[7], v_ada[8]) if self.cross_attention_adaln else None,
            )
            del vx_normed

        if run_ax:
            assert audio is not None and ax is not None
            a_ada = self._all_ada_values(self.audio_scale_shift_table, ax.shape[0], audio.timesteps)
            ashift_msa, ascale_msa, agate_msa = a_ada[0], a_ada[1], a_ada[2]

            norm_ax = self.ada_zero_function(ax, self.norm_eps, ascale_msa, ashift_msa)
            del ashift_msa, ascale_msa
            ax_msa_out = self.audio_attn1(
                norm_ax,
                pe=audio.positional_embeddings,
                mask=audio.self_attention_mask,
                perturbation_mask=audio.self_attn_perturbation_mask,
                all_perturbed=audio.self_attn_all_perturbed,
                cache=audio.attention_cache,
            )
            ax, ax_normed = self.post_sa_function(ax, ax_msa_out, None, self.norm_eps, agate_msa)
            del agate_msa, norm_ax, ax_msa_out
            ax = ax + self._apply_text_cross_attention(
                ax_normed,
                audio.context,
                self.audio_attn2,
                self.audio_scale_shift_table,
                getattr(self, "audio_prompt_scale_shift_table", None),
                audio.timesteps,
                audio.prompt_timestep,
                audio.context_mask,
                cross_attention_adaln=self.cross_attention_adaln,
                ca_ada_values=(a_ada[6], a_ada[7], a_ada[8]) if self.cross_attention_adaln else None,
            )
            del ax_normed

        # Audio - Video cross attention.
        if run_a2v or run_v2a:
            assert video is not None and audio is not None and vx is not None and ax is not None
            # Snapshot vx/ax before A2V mutates vx; V2A's video keys/values must
            # use the pre-A2V state so direction order doesn't bias the result.
            vx_pre_av = vx
            ax_pre_av = ax
            if run_a2v and not video.cross_attn_skip_all:
                scale_ca_video_a2v, shift_ca_video_a2v, gate_out_a2v = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                    slice(0, 2),
                )
                a2v_vx_scaled = self.ada_zero_function(vx_pre_av, self.norm_eps, scale_ca_video_a2v, shift_ca_video_a2v)
                del scale_ca_video_a2v, shift_ca_video_a2v

                scale_ca_audio_a2v, shift_ca_audio_a2v, _ = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                    slice(0, 2),
                )
                a2v_ax_scaled = self.ada_zero_function(ax_pre_av, self.norm_eps, scale_ca_audio_a2v, shift_ca_audio_a2v)
                del scale_ca_audio_a2v, shift_ca_audio_a2v
                a2v_attn_out = self.audio_to_video_attn(
                    a2v_vx_scaled,
                    context=a2v_ax_scaled,
                    pe=video.cross_positional_embeddings,
                    k_pe=audio.cross_positional_embeddings,
                    cache=video.attention_cache,
                )
                if video.cross_attn_perturbation_mask is not None:
                    vx = vx + a2v_attn_out * gate_out_a2v * video.cross_attn_perturbation_mask
                else:
                    vx = self._add_gated_residual(vx, a2v_attn_out, gate_out_a2v)
                del gate_out_a2v, a2v_vx_scaled, a2v_ax_scaled, a2v_attn_out

            if run_v2a and not audio.cross_attn_skip_all:
                scale_ca_audio_v2a, shift_ca_audio_v2a, gate_out_v2a = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                    slice(2, 4),
                )
                v2a_ax_scaled = self.ada_zero_function(ax_pre_av, self.norm_eps, scale_ca_audio_v2a, shift_ca_audio_v2a)
                del scale_ca_audio_v2a, shift_ca_audio_v2a
                scale_ca_video_v2a, shift_ca_video_v2a, _ = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                    slice(2, 4),
                )
                v2a_vx_scaled = self.ada_zero_function(vx_pre_av, self.norm_eps, scale_ca_video_v2a, shift_ca_video_v2a)
                del scale_ca_video_v2a, shift_ca_video_v2a
                v2a_attn_out = self.video_to_audio_attn(
                    v2a_ax_scaled,
                    context=v2a_vx_scaled,
                    pe=audio.cross_positional_embeddings,
                    k_pe=video.cross_positional_embeddings,
                    cache=audio.attention_cache,
                )
                if audio.cross_attn_perturbation_mask is not None:
                    ax = ax + v2a_attn_out * gate_out_v2a * audio.cross_attn_perturbation_mask
                else:
                    ax = self._add_gated_residual(ax, v2a_attn_out, gate_out_v2a)
                del gate_out_v2a, v2a_vx_scaled, v2a_ax_scaled, v2a_attn_out
            del vx_pre_av, ax_pre_av

        if run_vx:
            assert v_ada is not None and vx is not None
            vshift_mlp, vscale_mlp, vgate_mlp = v_ada[3], v_ada[4], v_ada[5]
            vx_scaled = self.ada_zero_function(vx, self.norm_eps, vscale_mlp, vshift_mlp)
            vx = self._add_gated_residual(vx, self.ff(vx_scaled), vgate_mlp)

            del vshift_mlp, vscale_mlp, vgate_mlp, vx_scaled

        if run_ax:
            assert a_ada is not None and ax is not None
            ashift_mlp, ascale_mlp, agate_mlp = a_ada[3], a_ada[4], a_ada[5]
            ax_scaled = self.ada_zero_function(ax, self.norm_eps, ascale_mlp, ashift_mlp)
            ax = self._add_gated_residual(ax, self.audio_ff(ax_scaled), agate_mlp)

            del ashift_mlp, ascale_mlp, agate_mlp, ax_scaled

        return replace(video, x=vx) if video is not None else None, replace(audio, x=ax) if audio is not None else None


def apply_cross_attention_adaln(
    x: torch.Tensor,
    context: torch.Tensor,
    attn: AttentionCallable,
    q_shift: torch.Tensor,
    q_scale: torch.Tensor,
    q_gate: torch.Tensor,
    prompt_scale_shift_table: torch.Tensor,
    prompt_timestep: torch.Tensor | None,
    context_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Modulate normalized queries and context with static or timestep-conditioned AdaLN."""
    batch_size = x.shape[0]
    kv_modulation = prompt_scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
    if prompt_timestep is not None:
        kv_modulation = kv_modulation + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
    shift_kv, scale_kv = kv_modulation.unbind(dim=2)
    attn_input = x * (1 + q_scale) + q_shift
    encoder_hidden_states = context * (1 + scale_kv) + shift_kv
    return attn(attn_input, context=encoder_hidden_states, mask=context_mask) * q_gate
