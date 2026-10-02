"""AlayaWorld conditioning on top of the canonical in-tree LTX-2.3 model.

Only Alaya-specific action and prefix-memory behavior lives here.  Transformer
blocks, RoPE, AdaLN, text cross attention, attention-kernel dispatch and model
configuration are inherited from :mod:`ltx_core`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from worldfoundry.base_models.diffusion_model.loaders import (
    CheckpointSpec,
    ModuleLoadSpec,
    NativeModuleLoader,
    safetensors_json_metadata,
)
from worldfoundry.base_models.diffusion_model.models.denoisers.ltx_configurator import (
    LTXVideoOnlyModelConfigurator,
)
from worldfoundry.base_models.diffusion_model.models.networks.ltx.modality import Modality
from worldfoundry.base_models.diffusion_model.models.networks.ltx.model import LTXModel
from worldfoundry.base_models.diffusion_model.models.networks.ltx.transformer_args import TransformerArgs
from worldfoundry.base_models.diffusion_model.optimizations import RuntimePolicy
from worldfoundry.core.attention.parallel.ulysses_attention import flattened_ulysses_attention
from worldfoundry.core.distributed.model_parallel.context import cat_outputs_cp, split_inputs_cp


@dataclass(frozen=True)
class AlayaPrefix:
    """One clean prefix stream consumed by AlayaWorld self attention.

    Exactly one of ``latent`` and ``tokens`` must be supplied. ``latent`` is
    already LTX-patchified with shape ``[B, N, 128]``; ``tokens`` is an
    Alaya-history embedding with shape ``[B, N, 4096]``. ``positions`` follows
    :class:`Modality` and is already converted to pixel/fps coordinates.
    """

    positions: torch.Tensor
    latent: torch.Tensor | None = None
    tokens: torch.Tensor | None = None
    valid_mask: torch.Tensor | None = None
    actions: torch.Tensor | None = None


@dataclass(frozen=True)
class AlayaConditioning:
    """Named Alaya prefix streams in their checkpoint-trained order."""

    sink: AlayaPrefix | None = None
    history: AlayaPrefix | None = None
    spatial: AlayaPrefix | None = None
    nearby: AlayaPrefix | None = None
    target_actions: torch.Tensor | None = None
    target_frames: int | None = None

    def prefixes(self) -> tuple[AlayaPrefix, ...]:
        return tuple(value for value in (self.sink, self.history, self.spatial, self.nearby) if value is not None)


class ActionAdaLNEmbedder(nn.Module):
    """Embed six scaled camera-pose deltas using AlayaWorld's checkpoint ABI."""

    def __init__(
        self,
        dim: int,
        action_dim: int = 6,
        subframes: int = 8,
        freq_dim_per_axis: int = 32,
        freq_scale: float = 1000.0,
    ) -> None:
        super().__init__()
        if freq_dim_per_axis <= 0 or freq_dim_per_axis % 2:
            raise ValueError("freq_dim_per_axis must be a positive even integer")
        self.action_dim = int(action_dim)
        self.subframes = int(subframes)
        self.freq_dim_per_axis = int(freq_dim_per_axis)
        self.freq_scale = float(freq_scale)
        self.mlp = nn.Sequential(
            nn.Linear(self.action_dim * self.freq_dim_per_axis, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    @staticmethod
    def _sinusoidal_embedding(dim: int, position: torch.Tensor) -> torch.Tensor:
        original_shape = position.shape
        flattened = position.flatten().to(torch.float64)
        frequencies = torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(max(dim // 2, 1)),
        )
        sinusoid = torch.outer(flattened, frequencies)
        embedding = torch.cat((torch.cos(sinusoid), torch.sin(sinusoid)), dim=-1)
        return embedding.view(*original_shape, dim).to(position.dtype)

    def forward(self, action_vectors: torch.Tensor) -> torch.Tensor:
        if action_vectors.ndim == 2:
            action_vectors = action_vectors.unsqueeze(0)
        if action_vectors.ndim != 3 or action_vectors.shape[-1] != self.action_dim:
            raise ValueError(f"action vectors must be [B,T,{self.action_dim}], got {tuple(action_vectors.shape)}")
        embeddings = tuple(
            self._sinusoidal_embedding(
                self.freq_dim_per_axis,
                action_vectors[..., axis] * self.freq_scale,
            )
            for axis in range(self.action_dim)
        )
        return self.mlp(torch.cat(embeddings, dim=-1).to(self.mlp[0].weight.dtype))


class AlayaWorldModel(LTXModel):
    """LTX-2.3 video model augmented with Alaya action and memory prefixes."""

    def __init__(
        self,
        *args,
        action_dim: int = 6,
        action_subframes: int = 8,
        action_freq_dim_per_axis: int = 32,
        action_freq_scale: float = 1000.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.action_adaln_embedder = ActionAdaLNEmbedder(
            dim=self.inner_dim,
            action_dim=action_dim,
            subframes=action_subframes,
            freq_dim_per_axis=action_freq_dim_per_axis,
            freq_scale=action_freq_scale,
        )
        self.action_adaln_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.inner_dim, self._adaln_embedding_coefficient * self.inner_dim, bias=True),
        )
        self._context_parallel_group = None

    def enable_context_parallel(self, group=None) -> "AlayaWorldModel":
        """Enable Ulysses self attention using an initialized process group.

        The existing per-device attention implementation remains the inner
        kernel. Consequently Ampere/Ada use their supported SDPA path while
        Hopper and Blackwell may select newer fused kernels without an Alaya
        hard dependency on FlashAttention-3.
        """

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before enabling context parallelism")
        group = dist.group.WORLD if group is None else group
        world_size = dist.get_world_size(group)
        if world_size <= 1:
            self._context_parallel_group = None
            return self
        if self.num_attention_heads % world_size:
            raise ValueError(f"AlayaWorld has {self.num_attention_heads} heads, not divisible by CP size {world_size}")
        if self._context_parallel_group is not None:
            if self._context_parallel_group is group:
                return self
            raise RuntimeError("context parallelism is already configured with another process group")

        for block in self.transformer_blocks:
            attention = block.attn1
            unmasked = attention.attention_function
            masked = attention.masked_attention_function

            def _unmasked(q, k, v, heads, *, _inner=unmasked, _group=group):
                return flattened_ulysses_attention(
                    q,
                    k,
                    v,
                    heads,
                    attention_fn=_inner,
                    group=_group,
                )

            def _masked(q, k, v, heads, mask, *, _inner=masked, _group=group):
                return flattened_ulysses_attention(
                    q,
                    k,
                    v,
                    heads,
                    attention_fn=_inner,
                    mask=mask,
                    group=_group,
                )

            attention.attention_function = _unmasked
            attention.masked_attention_function = _masked
        self._context_parallel_group = group
        return self

    @staticmethod
    def _check_prefix(prefix: AlayaPrefix, batch: int) -> int:
        if (prefix.latent is None) == (prefix.tokens is None):
            raise ValueError("each Alaya prefix must provide exactly one of latent or tokens")
        value = prefix.latent if prefix.latent is not None else prefix.tokens
        if value.ndim != 3 or value.shape[0] != batch:
            raise ValueError(f"prefix values must be [B,N,C] with B={batch}, got {tuple(value.shape)}")
        token_count = value.shape[1]
        if prefix.positions.ndim != 4 or prefix.positions.shape[:3] != (batch, 3, token_count):
            raise ValueError(
                f"prefix positions must be [B,3,N,2], got {tuple(prefix.positions.shape)} for {token_count} tokens"
            )
        if prefix.positions.shape[-1] != 2:
            raise ValueError(f"prefix position bounds must have size 2, got {tuple(prefix.positions.shape)}")
        if prefix.valid_mask is not None and prefix.valid_mask.shape != (batch, token_count):
            raise ValueError(
                f"prefix valid_mask must be [B,N]={batch, token_count}, got {tuple(prefix.valid_mask.shape)}"
            )
        return token_count

    def _action_tokens(
        self,
        actions: torch.Tensor,
        token_count: int,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = actions.to(device=device, dtype=dtype)
        if values.ndim == 2:
            values = values.unsqueeze(0)
        if values.ndim != 3 or values.shape[-1] != 6 or values.shape[1] == 0:
            raise ValueError(f"actions must be non-empty [B,T,6], got {tuple(values.shape)}")
        if values.shape[0] == 1 and batch > 1:
            values = values.expand(batch, -1, -1)
        elif values.shape[0] != batch:
            raise ValueError(f"action batch {values.shape[0]} does not match latent batch {batch}")

        embedded = self.action_adaln_embedder(values)
        projected = self.action_adaln_projection(embedded)
        frame_count = values.shape[1]
        if token_count % frame_count == 0:
            repeats = token_count // frame_count
            return (
                embedded.repeat_interleave(repeats, dim=1),
                projected.repeat_interleave(repeats, dim=1),
            )

        def _resize(value: torch.Tensor) -> torch.Tensor:
            source_dtype = value.dtype
            value = F.interpolate(
                value.transpose(1, 2).float(),
                size=token_count,
                mode="linear",
                align_corners=True,
            )
            return value.transpose(1, 2).to(source_dtype)

        return _resize(embedded), _resize(projected)

    def _add_action_conditioning(
        self,
        args: TransformerArgs,
        conditioning: AlayaConditioning,
        prefixes: tuple[AlayaPrefix, ...],
        prefix_counts: tuple[int, ...],
        target_count: int,
    ) -> TransformerArgs:
        target_actions = conditioning.target_actions
        if target_actions is None:
            return args
        batch = args.x.shape[0]
        action_embedded, action_projected = self._action_tokens(
            target_actions,
            target_count,
            batch=batch,
            device=args.x.device,
            dtype=args.x.dtype,
        )
        embedded_parts: list[torch.Tensor] = []
        projected_parts: list[torch.Tensor] = []
        for prefix, count in zip(prefixes, prefix_counts, strict=True):
            if prefix.actions is None:
                embedded_parts.append(action_embedded.new_zeros(batch, count, action_embedded.shape[-1]))
                projected_parts.append(action_projected.new_zeros(batch, count, action_projected.shape[-1]))
            else:
                prefix_embedded, prefix_projected = self._action_tokens(
                    prefix.actions,
                    count,
                    batch=batch,
                    device=args.x.device,
                    dtype=args.x.dtype,
                )
                embedded_parts.append(prefix_embedded)
                projected_parts.append(prefix_projected)
        embedded_parts.append(action_embedded)
        projected_parts.append(action_projected)
        return replace(
            args,
            embedded_timestep=args.embedded_timestep
            + torch.cat(embedded_parts, dim=1).to(args.embedded_timestep.dtype),
            timesteps=args.timesteps + torch.cat(projected_parts, dim=1).to(args.timesteps.dtype),
        )

    def _prepend_conditioning(
        self,
        args: TransformerArgs,
        video: Modality,
        conditioning: AlayaConditioning,
    ) -> tuple[TransformerArgs, int, int]:
        prefixes = conditioning.prefixes()
        if not prefixes:
            return self._add_action_conditioning(args, conditioning, (), (), args.x.shape[1]), 0, args.x.shape[1]

        batch, target_count = args.x.shape[:2]
        prefix_counts = tuple(self._check_prefix(prefix, batch) for prefix in prefixes)
        prefix_tokens = tuple(
            self.patchify_proj(prefix.latent.to(device=args.x.device, dtype=args.x.dtype))
            if prefix.latent is not None
            else prefix.tokens.to(device=args.x.device, dtype=args.x.dtype)
            for prefix in prefixes
        )
        x = torch.cat((*prefix_tokens, args.x), dim=1)
        positions = torch.cat(
            tuple(
                prefix.positions.to(device=video.positions.device, dtype=video.positions.dtype) for prefix in prefixes
            )
            + (video.positions,),
            dim=2,
        )
        positional_embeddings = self.video_args_preprocessor._prepare_positional_embeddings(
            positions=positions,
            inner_dim=self.inner_dim,
            max_pos=self.positional_embedding_max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.num_attention_heads,
            x_dtype=args.x.dtype,
        )

        zero_sigma = torch.zeros((batch, 1), device=args.x.device, dtype=video.timesteps.dtype)
        zero_timestep, zero_embedded = self.video_args_preprocessor._prepare_timestep(
            zero_sigma,
            self.adaln_single,
            batch,
            args.x.dtype,
        )
        total_prefix = sum(prefix_counts)
        timesteps = torch.cat((zero_timestep.expand(batch, total_prefix, -1), args.timesteps), dim=1)
        embedded_timestep = torch.cat(
            (zero_embedded.expand(batch, total_prefix, -1), args.embedded_timestep),
            dim=1,
        )

        any_mask = any(prefix.valid_mask is not None for prefix in prefixes)
        self_attention_mask = args.self_attention_mask
        if any_mask:
            if self_attention_mask is not None:
                raise ValueError("combining a target dense attention mask with Alaya prefix masks is unsupported")
            valid_parts = tuple(
                prefix.valid_mask.to(device=args.x.device, dtype=torch.bool)
                if prefix.valid_mask is not None
                else torch.ones(batch, count, device=args.x.device, dtype=torch.bool)
                for prefix, count in zip(prefixes, prefix_counts, strict=True)
            )
            valid = torch.cat(
                (*valid_parts, torch.ones(batch, target_count, device=args.x.device, dtype=torch.bool)), dim=1
            )
            if not bool(valid.all()):
                bias = torch.zeros(batch, 1, 1, valid.shape[1], device=args.x.device, dtype=args.x.dtype)
                self_attention_mask = bias.masked_fill(~valid[:, None, None], -10000.0)

        args = replace(
            args,
            x=x,
            timesteps=timesteps,
            embedded_timestep=embedded_timestep,
            positional_embeddings=positional_embeddings,
            self_attention_mask=self_attention_mask,
        )
        args = self._add_action_conditioning(args, conditioning, prefixes, prefix_counts, target_count)
        return args, total_prefix, target_count

    def _pad_and_split_cp(self, args: TransformerArgs) -> tuple[TransformerArgs, int]:
        group = self._context_parallel_group
        if group is None:
            return args, args.x.shape[1]
        world_size = dist.get_world_size(group)
        original_length = args.x.shape[1]
        padding = (-original_length) % world_size
        if padding:
            if args.self_attention_mask is None:
                self_attention_mask = torch.zeros(
                    args.x.shape[0],
                    1,
                    1,
                    original_length + padding,
                    device=args.x.device,
                    dtype=args.x.dtype,
                )
                self_attention_mask[..., original_length:] = -10000.0
            else:
                self_attention_mask = F.pad(args.self_attention_mask, (0, padding), value=-10000.0)
            args = replace(
                args,
                x=F.pad(args.x, (0, 0, 0, padding)),
                timesteps=F.pad(args.timesteps, (0, 0, 0, padding)),
                embedded_timestep=F.pad(args.embedded_timestep, (0, 0, 0, padding)),
                positional_embeddings=tuple(F.pad(value, (0, 0, 0, padding)) for value in args.positional_embeddings),
                self_attention_mask=self_attention_mask,
            )

        args = replace(
            args,
            x=split_inputs_cp(args.x, 1, group),
            timesteps=split_inputs_cp(args.timesteps, 1, group),
            embedded_timestep=split_inputs_cp(args.embedded_timestep, 1, group),
            positional_embeddings=tuple(split_inputs_cp(value, 2, group) for value in args.positional_embeddings),
        )
        return args, original_length

    def _gather_cp(self, args: TransformerArgs, original_length: int) -> TransformerArgs:
        group = self._context_parallel_group
        if group is None:
            return args
        return replace(
            args,
            x=cat_outputs_cp(args.x, 1, group)[:, :original_length],
            embedded_timestep=cat_outputs_cp(args.embedded_timestep, 1, group)[:, :original_length],
        )

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None = None,
        perturbations=None,
        conditioning: AlayaConditioning | None = None,
    ) -> tuple[torch.Tensor | None, None]:
        if video is None:
            return None, None
        if audio is not None:
            raise ValueError("AlayaWorld is an LTX video-only model")
        args = self.video_args_preprocessor.prepare(video, None)
        prefix_count = 0
        target_count = args.x.shape[1]
        if conditioning is not None:
            args, prefix_count, target_count = self._prepend_conditioning(args, video, conditioning)

        args, cp_original_length = self._pad_and_split_cp(args)
        output, _ = self._process_transformer_blocks(args, None, perturbations)
        output = self._gather_cp(output, cp_original_length)
        if prefix_count:
            output = replace(
                output,
                x=output.x[:, prefix_count : prefix_count + target_count].contiguous(),
                embedded_timestep=output.embedded_timestep[:, prefix_count : prefix_count + target_count].contiguous(),
            )
        velocity = self._process_output(
            self.scale_shift_table,
            self.norm_out,
            self.proj_out,
            output.x,
            output.embedded_timestep,
        )
        return velocity, None


class AlayaWorldModelConfigurator(LTXVideoOnlyModelConfigurator):
    """Build the thin Alaya variant from an LTX-2.3 safetensors config."""

    MODEL_CLS = AlayaWorldModel


class AlayaWorldTransformerModule(nn.Module):
    """Checkpoint-configured Alaya transformer loaded by the shared infra."""

    def __init__(self, checkpoint_config: dict[str, object]) -> None:
        super().__init__()
        self.model = AlayaWorldModelConfigurator.from_config(checkpoint_config)


_ALAYA_TRANSFORMER_ROOTS = (
    "blocks.",
    "patchify_proj.",
    "adaln_single.",
    "prompt_adaln_single.",
    "scale_shift_table",
    "proj_out.",
    "action_adaln_embedder.",
    "action_adaln_projection.",
)


def convert_alaya_transformer_state_dict(state_dict):
    """Select Alaya transformer tensors and map the released block prefix."""

    converted = {}
    for key, value in state_dict.items():
        if not key.startswith(_ALAYA_TRANSFORMER_ROOTS):
            continue
        destination = key.replace("blocks.", "transformer_blocks.", 1) if key.startswith("blocks.") else key
        converted[f"model.{destination}"] = value
    return converted


def load_alaya_world_model(
    checkpoint: CheckpointSpec,
    policy: RuntimePolicy,
) -> AlayaWorldModel:
    """Load AlayaWorld through the same native loader used by every LTX model."""

    wrapper = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=AlayaWorldTransformerModule,
            config_resolver=lambda materialized: {"checkpoint_config": dict(safetensors_json_metadata(materialized))},
            state_dict_converter=convert_alaya_transformer_state_dict,
            layer_container="model.transformer_blocks",
        ),
        checkpoint,
        policy,
    )
    if not isinstance(wrapper, AlayaWorldTransformerModule):
        raise TypeError(f"expected AlayaWorldTransformerModule, got {type(wrapper).__name__}")
    return wrapper.model


__all__ = [
    "ActionAdaLNEmbedder",
    "AlayaConditioning",
    "AlayaPrefix",
    "AlayaWorldModel",
    "AlayaWorldModelConfigurator",
    "AlayaWorldTransformerModule",
    "convert_alaya_transformer_state_dict",
    "load_alaya_world_model",
]
