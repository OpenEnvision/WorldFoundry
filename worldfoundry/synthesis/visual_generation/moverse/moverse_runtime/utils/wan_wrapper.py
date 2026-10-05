import os
from typing import List, Optional

import torch

from worldfoundry.base_models.diffusion_model.schedulers import (
    InferenceFlowMatchScheduler,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.resident import (
    WanVAEWrapper as _WanVAEWrapper,
)
from worldfoundry.base_models.diffusion_model.models.encoders.wan.resident import (
    WanTextEncoder as _WanTextEncoder,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.moverse.causal import (
    CausalWanModel,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.moverse.mmdit import (
    CausalMMDiTModel,
)
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.moverse.causal import (
    causal_rope_apply_shared_height,
    parse_memrope_cfg,
)
from worldfoundry.base_models.diffusion_model.loaders.wan_variant import (
    load_wan_config,
    load_wan_transformer,
)


def _wan_model_path(model_name: str) -> str:
    """Resolve Wan assets from the WorldFoundry runtime environment."""

    configured = os.environ.get("WAN_MODEL_PATH")
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    return os.path.join("checkpoints", model_name)


class WanTextEncoder(_WanTextEncoder):
    def __init__(self) -> None:
        super().__init__(_wan_model_path("Wan2.1-T2V-1.3B"))


class WanVAEWrapper(_WanVAEWrapper):
    def __init__(self) -> None:
        super().__init__(_wan_model_path("Wan2.1-T2V-1.3B"))


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0,
            token_concat_and_share_rope=False,
            recam_concat=False,
            skip_pretrained=False,
            is_mmdit=False,
            double_stream_layers=None,
            cross_attn_norm=True,
            fixed_timestep=-1,
            online_rope_indexing=False,
            memrope=None,
            rope_precision="fp64",
    ):
        """
        Args:
            token_concat_and_share_rope (bool): Usually supplied through config kwarg.
                When True, injects a shared-height RoPE variant into every attention
                block via set_rope_fn() so that height-doubled inputs (noisy | cond
                concatenated along H) receive shared positional embeddings. Must be
                set to True when passing cond_latent to forward() in height-concat mode.
                Has no effect when recam_concat=True.
            recam_concat (bool): When True, uses ReCamMaster-style frame-dimension
                concatenation: cond_latent is concatenated with the noisy input along
                the frame dimension (dim=2 in C-first layout), producing a
                [B, C, 2F, H, W] input to the DiT. Standard RoPE is used (frames
                0..2F-1 each receive their own position), matching rope_shared_frame=False
                in the original ReCamMaster pipeline. The output is sliced back to the
                first F frames (the denoised prediction). No rope manipulation is applied.
                When False (default), height-concat mode is used if cond_latent is given.
            skip_pretrained (bool): When True, initialise the model architecture from
                the saved config only (no weights loaded from disk). Use this when a
                checkpoint will be loaded on top immediately after construction, to
                avoid the cost of loading pretrained weights that would be overwritten.
            is_mmdit (bool): When True, use CausalMMDiTModel as the generator.
                cond_latent is passed as a separate stream (no height-concat).
                Requires is_causal=True.  frame_seq_length stays at 1560 (no doubling).
            double_stream_layers (list[int], optional): Block indices for dual-stream
                processing in CausalMMDiTModel.  Forwarded to CausalMMDiTModel constructor.
                Default: [0, 1, 4, 8, 12, 16, 20, 24, 28, 29].
            cross_attn_norm (bool): Whether to use WanLayerNorm (with learnable weight/bias)
                before cross-attention (norm3 / cond_norm3).  Wan2.1-T2V-1.3B was trained
                with cross_attn_norm=True, so checkpoints initialised from it need this True.
                Default: False (matches CausalMMDiTModel default; set True for Wan 1.3B).
            fixed_timestep (float): If >= 0, override the scheduler timestep with this fixed value at every forward pass.
        """
        super().__init__()

        if model_name == "Wan2.1-T2V-1.3B":
            model_path = _wan_model_path(model_name)
            print(f"Using model path: {model_path}")
        elif model_name == "Wan2.1-T2V-14B":
            model_path = _wan_model_path(model_name)
        else:
            raise ValueError(f"Unsupported Wan model: {model_name}")

        if not is_causal:
            raise ValueError("MoVerse only bundles its causal inference model")

        self.is_mmdit = is_mmdit

        if skip_pretrained:
            # Load architecture config only — no pretrained weights.
            # The caller is responsible for loading a checkpoint afterwards.
            if self.is_mmdit:
                _double = double_stream_layers if double_stream_layers is not None else [0, 1, 4, 8, 12, 16, 20, 24, 28, 29]
                self.model = CausalMMDiTModel(
                    local_attn_size=local_attn_size, sink_size=sink_size,
                    double_stream_layers=_double, cross_attn_norm=cross_attn_norm)
            else:
                cfg = load_wan_config(model_path)
                self.model = CausalWanModel.from_config(
                    cfg, local_attn_size=local_attn_size, sink_size=sink_size)
            print(f"Initialised {model_name} architecture only (skip_pretrained=True).")
        else:
            if self.is_mmdit:
                _double = double_stream_layers if double_stream_layers is not None else [0, 1, 4, 8, 12, 16, 20, 24, 28, 29]
                # Build from pretrained CausalWanModel, then load weights via MMDiT loader
                self.model = CausalMMDiTModel(
                    local_attn_size=local_attn_size, sink_size=sink_size,
                    double_stream_layers=_double, cross_attn_norm=cross_attn_norm)
                print(f"CausalMMDiTModel initialised (weights must be loaded separately via load_causal_mmdit_weights).")
            else:
                self.model = load_wan_transformer(
                    CausalWanModel,
                    model_path,
                    additional_kwargs={
                        "local_attn_size": local_attn_size,
                        "sink_size": sink_size,
                    },
                )
        self.model.eval()

        self.fixed_timestep = fixed_timestep

        self.scheduler = InferenceFlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000)

        # Default seq_len for [B, F=21, C=16, H=60, W=104]; recomputed dynamically
        # when cond_latent is provided (height doubles → seq_len doubles).
        self.seq_len = 32760  # [1, 21, 16, 60, 104]

        self.recam_concat = recam_concat

        # Inject shared-height RoPE variants as instance-level dependencies so
        # that different model instances remain isolated (no module-level patching).
        # recam_concat uses standard RoPE over the doubled frame sequence — no injection needed.
        if token_concat_and_share_rope and not recam_concat:
            self.model.set_rope_fn(causal_rope_fn=causal_rope_apply_shared_height)
            print(f"Applied shared-height RoPE variant via set_rope_fn().")
        elif recam_concat:
            print("recam_concat=True: frame-dim concat, standard RoPE (no injection).")

        # MemRoPE Phase 1 — Online RoPE Indexing (height-concat causal).
        # Re-indexes RoPE block-relative in the AR/KV-cache path so indices stay within the
        # trained range for >1024-frame rollouts.
        self.online_rope_indexing = online_rope_indexing
        if online_rope_indexing:
            if not self.is_mmdit and hasattr(self.model, "set_online_rope"):
                self.model.set_online_rope(True)
                print("Enabled MemRoPE Online RoPE Indexing (set_online_rope=True).")
            else:
                raise ValueError(
                    "online_rope_indexing=True requires is_causal=True and non-MMDiT model "
                    f"(got is_causal={is_causal}, is_mmdit={self.is_mmdit}).")

        # MemRoPE Phase 2 — dual-EMA Memory Tokens with per-half (gen/cond) three-tier cache.
        # Requires height-concat conditioning (token_concat_and_share_rope). Supersedes the
        # online_rope_indexing path. See SUMMARY_MEMROPE.md.
        self.memrope_cfg = parse_memrope_cfg(memrope)
        if self.memrope_cfg is not None:
            if not (not self.is_mmdit and token_concat_and_share_rope
                    and hasattr(self.model, "set_memrope")):
                raise ValueError(
                    "memrope requires is_causal=True, non-MMDiT, and "
                    "token_concat_and_share_rope=True (height-concat conditioning). "
                    f"Got is_causal={is_causal}, is_mmdit={self.is_mmdit}, "
                    f"token_concat_and_share_rope={token_concat_and_share_rope}.")
            self.model.set_memrope(self.memrope_cfg)
            print(f"Enabled MemRoPE (set_memrope): fused={self.memrope_cfg['fused']} "
                  f"gen={self.memrope_cfg['gen']} cond={self.memrope_cfg['cond']}.")

        # RoPE compute precision for the AR inference rope calls (online + memrope).
        # fp64 (default) = exact; fp32 ≈ 2x faster rope, tiny precision change.
        if not self.is_mmdit and hasattr(self.model, "set_rope_precision"):
            self.model.set_rope_precision(rope_precision)
        if str(rope_precision).lower() not in ("fp64", "float64"):
            print(f"RoPE compute precision set to {rope_precision}.")

        print(f"WanDiffusionWrapper for {model_name} initialised with is_causal={is_causal}, recam_concat={recam_concat}, token_concat_and_share_rope={token_concat_and_share_rope}, is_mmdit={self.is_mmdit}."
              f" local_attn_size={local_attn_size}, sink_size={sink_size}, cross_attn_norm={cross_attn_norm}, fixed_timestep={fixed_timestep}, online_rope_indexing={online_rope_indexing}.")

    def _compute_seq_len(self, model_input_c_first: torch.Tensor) -> int:
        """
        Compute the transformer sequence length (number of tokens) from a C-first input tensor.

        Args:
            model_input_c_first: Tensor of shape [B, C, F, H, W] as fed to self.model.
        Returns:
            seq_len (int): F_patches * H_patches * W_patches
        """
        _, _, F, H, W = model_input_c_first.shape
        pt, ph, pw = self.model.patch_size
        return (F // pt) * (H // ph) * (W // pw)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        cond_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            cond_latent (Tensor, optional): Conditioning render/reference latent of shape
                [B, F_cond, C, H, W]. Concatenation mode depends on recam_concat:
                - recam_concat=True (ReCamMaster-style): concatenated along the frame
                  dimension → [B, C, F+F_cond, H, W]. Standard RoPE assigns unique
                  positions to all frames. Output is sliced back to the first F frames.
                - recam_concat=False (height-concat): concatenated along height dim →
                  [B, C, F, H+H_cond, W]. Requires token_concat_and_share_rope=True so
                  both halves share height positional embeddings. Output sliced to top half.
        """
        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.fixed_timestep >= 0:
            input_timestep = torch.full((timestep.shape[0],), self.fixed_timestep, device=timestep.device, dtype=timestep.dtype)
        else:
            input_timestep = timestep

        # Permute noisy input from [B, F, C, H, W] to [B, C, F, H, W]
        noisy_c_first = noisy_image_or_video.permute(0, 2, 1, 3, 4)
        F_noisy = noisy_c_first.shape[2]  # original latent frame count
        H_noisy = noisy_c_first.shape[3]  # original latent height

        if cond_latent is not None and self.is_mmdit:
            # MMDiT mode: cond_latent passed as a separate stream — no concat
            cond_x = cond_latent.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, F, H, W]
            model_input = noisy_c_first
            seq_len = self.seq_len
        elif cond_latent is not None:
            cond_c_first = cond_latent.permute(0, 2, 1, 3, 4)  # [B, C, F_cond, H, W]
            if self.recam_concat:
                # ReCamMaster-style: concatenate along frame dim → [B, C, F+F_cond, H, W]
                # Standard RoPE assigns positions 0..F+F_cond-1 (no rope manipulation).
                model_input = torch.cat([noisy_c_first, cond_c_first], dim=2)
            else:
                # Height-concat mode: [B, C, F, H+H_cond, W].  Slice cond along the
                # frame dim to noisy's temporal extent in case caller passed a longer
                # cond_latent (for example, a longer conditioning sequence).
                # Mirrors the existing slice in the clean_x branch below.
                assert cond_c_first.shape[2] >= F_noisy, (
                    f"cond_latent has fewer frames ({cond_c_first.shape[2]}) than "
                    f"noisy ({F_noisy}) — cannot height-concat."
                )
                cond_c_first = cond_c_first[:, :, :F_noisy]
                model_input = torch.cat([noisy_c_first, cond_c_first], dim=3)
            seq_len = self._compute_seq_len(model_input)
            cond_x = None
        else:
            model_input = noisy_c_first
            seq_len = self.seq_len
            cond_x = None

        if kv_cache is None:
            raise ValueError("MoVerse inference requires an initialized KV cache")
        flow_pred_c_first = self.model(
            model_input,
            t=input_timestep,
            context=prompt_embeds,
            seq_len=seq_len,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            **({"cond_x": cond_x} if self.is_mmdit else {}),
        )

        # When cond_latent was provided, slice output to keep only the noisy half.
        # MMDiT mode outputs x-stream only (no slicing needed).
        if cond_latent is not None and not self.is_mmdit:
            if self.recam_concat:
                flow_pred_c_first = flow_pred_c_first[:, :, :F_noisy, :, :]
            else:
                flow_pred_c_first = flow_pred_c_first[:, :, :, :H_noisy, :]

        # Permute output back to [B, F, C, H, W]
        flow_pred = flow_pred_c_first.permute(0, 2, 1, 3, 4)

        # pred_x0 is derived from the original (non-doubled) noisy input
        pred_x0 = self.scheduler.flow_to_x0(
            flow_pred.flatten(0, 1),
            noisy_image_or_video.flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])

        return flow_pred, pred_x0

    def get_scheduler(self) -> InferenceFlowMatchScheduler:
        return self.scheduler
