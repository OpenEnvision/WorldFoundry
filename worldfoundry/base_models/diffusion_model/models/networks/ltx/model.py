"""LTX velocity DiT and x0 wrappers.

:class:`LTXModel` is the checkpoint-shaped backbone.  Each modality arrives
as a :class:`Modality` of already-patchified ``B T D`` tokens (video:
spatial-temporal patches; audio: 1-D time).  Stem projections + AdaLN
preprocessors build :class:`TransformerArgs`; stacked
:class:`BasicAVTransformerBlock` layers run self-attn, text cross-attn,
optional A↔V cross-attn, and FFN.  The head is AdaLN scale-shift then a
linear map back to latent channels.

:class:`X0Model` converts predicted velocity to denoised latents using
per-token timesteps.  Sampling lives in the LTX denoiser, not here.
"""

import weakref
from collections.abc import Iterable, Mapping
from dataclasses import replace
from enum import Enum
from typing import Any

import torch

from worldfoundry.base_models.diffusion_model.models.networks.ltx.adaln import (
    AdaLayerNormSingle,
    adaln_embedding_coefficient,
)
from worldfoundry.base_models.diffusion_model.models.networks.ltx.modality import Modality
from worldfoundry.base_models.diffusion_model.models.networks.ltx.perturbations import (
    BatchedPerturbationConfig,
    PerturbationType,
)
from worldfoundry.base_models.diffusion_model.models.networks.ltx.rope import LTXRopeType
from worldfoundry.base_models.diffusion_model.models.networks.ltx.transformer import (
    BasicAVTransformerBlock,
    TransformerConfig,
)
from worldfoundry.base_models.diffusion_model.models.networks.ltx.transformer_args import (
    BlockPerturbationsProcessor,
    MultiModalTransformerArgsPreprocessor,
    TransformerArgs,
    TransformerArgsPreprocessor,
)
from worldfoundry.core.nn import DEFAULT_TRANSFORMER_OPS, TransformerOpsConfig
from worldfoundry.core.nn import velocity_to_denoised as to_denoised
from worldfoundry.core.nn.checkpointing.gradient import gradient_checkpoint_forward


def infer_video_grid_shape(positions: torch.Tensor) -> tuple[int, int, int] | None:
    """Infer exact ``(F, H, W)`` raster grid dimensions from LTX video ``positions``.

    Returns ``None`` when ``positions`` is not a 3-axis raster grid with
    ``F * H * W == T`` (for example when auxiliary reference tokens are appended).
    Call outside CUDA Graph capture so the single-step verification never syncs
    during capture or replay.
    """
    if positions.ndim == 4:
        coords = positions[0, :, :, 0]
    elif positions.ndim == 3:
        coords = positions[0]
    else:
        return None
    if coords.shape[0] != 3 or coords.shape[1] <= 0:
        return None
    total_tokens = int(coords.shape[1])
    f_count = int(torch.unique(coords[0]).numel())
    h_count = int(torch.unique(coords[1]).numel())
    w_count = int(torch.unique(coords[2]).numel())
    if f_count <= 0 or h_count <= 0 or w_count <= 0 or f_count * h_count * w_count != total_tokens:
        return None
    c0 = coords[0].view(f_count, h_count, w_count)
    c1 = coords[1].view(f_count, h_count, w_count)
    c2 = coords[2].view(f_count, h_count, w_count)
    raster_ok = (c0 == c0[:, :1, :1]).all() & (c1 == c1[:1, :, :1]).all() & (c2 == c2[:1, :1, :]).all()
    if positions.shape[0] > 1:
        raster_ok = raster_ok & (positions == positions[:1]).all()
    if not bool(raster_ok.item()):
        return None
    return (f_count, h_count, w_count)


def resolve_cached_video_grid_shape(
    cache: dict[int, tuple[weakref.ref[torch.Tensor], int, int, tuple[int, ...], tuple[int, int, int] | None]],
    positions: torch.Tensor,
) -> tuple[int, int, int] | None:
    """Return cached ``(F, H, W)`` for ``positions`` with zero host syncs on repeat steps."""
    key = id(positions)
    ptr = int(positions.data_ptr())
    version = -1 if positions.is_inference() else int(positions._version)
    shape = tuple(int(dim) for dim in positions.shape)
    entry = cache.get(key)
    if entry is not None:
        ref, cached_ptr, cached_ver, cached_shape, cached_grid = entry
        if ref() is positions and cached_ptr == ptr and cached_ver == version and cached_shape == shape:
            return cached_grid
    grid = infer_video_grid_shape(positions)
    if len(cache) >= 16:
        dead = [k for k, v in cache.items() if v[0]() is None]
        for k in dead:
            cache.pop(k, None)
        if len(cache) >= 16:
            cache.pop(next(iter(cache)))
    cache[key] = (weakref.ref(positions), ptr, version, shape, grid)
    return grid


def _build_tile_3d_permutation(
    f: int,
    h: int,
    w: int,
    tf: int,
    th: int,
    tw: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    """Build ``(perm, inv_perm, sink_tokens)`` grouping ``(F, H, W)`` tokens into 3D tiles."""
    f_full, h_full, w_full = (f // tf) * tf, (h // th) * th, (w // tw) * tw
    if f_full <= 0 or h_full <= 0 or w_full <= 0:
        return None
    total = f * h * w
    grid_idx = torch.arange(total, device=device, dtype=torch.long).view(f, h, w)
    core = (
        grid_idx[:f_full, :h_full, :w_full]
        .view(f_full // tf, tf, h_full // th, th, w_full // tw, tw)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(-1)
    )
    if core.numel() < total:
        in_core = torch.zeros(total, device=device, dtype=torch.bool)
        in_core[core] = True
        tail = torch.nonzero(~in_core, as_tuple=False).squeeze(-1)
        perm = torch.cat((tail, core), dim=0)
        sink_tokens = int(tail.numel())
    else:
        perm = core
        sink_tokens = 0
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(total, device=device, dtype=torch.long)
    return perm, inv_perm, sink_tokens


def _permute_rope(
    pe: tuple[torch.Tensor, torch.Tensor] | None,
    perm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if pe is None:
        return None
    dim = 2 if pe[0].ndim == 4 else 1
    return (pe[0].index_select(dim, perm), pe[1].index_select(dim, perm))


def _permute_video_args(video_args: TransformerArgs, perm: torch.Tensor) -> TransformerArgs:
    """Permute every per-token field of ``video_args`` by ``perm``."""
    token_count = int(video_args.x.shape[1])

    def _sel(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is not None and tensor.ndim >= 2 and tensor.shape[1] == token_count:
            return tensor.index_select(1, perm)
        return tensor

    context_mask = video_args.context_mask
    if context_mask is not None and context_mask.ndim == 4 and context_mask.shape[2] == token_count:
        context_mask = context_mask.index_select(2, perm)

    return replace(
        video_args,
        x=video_args.x.index_select(1, perm),
        context_mask=context_mask,
        timesteps=video_args.timesteps.index_select(1, perm)
        if video_args.timesteps.shape[1] == token_count
        else video_args.timesteps,
        embedded_timestep=video_args.embedded_timestep.index_select(1, perm)
        if video_args.embedded_timestep.shape[1] == token_count
        else video_args.embedded_timestep,
        positional_embeddings=_permute_rope(video_args.positional_embeddings, perm),  # type: ignore[arg-type]
        cross_positional_embeddings=_permute_rope(video_args.cross_positional_embeddings, perm),
        cross_scale_shift_timestep=_sel(video_args.cross_scale_shift_timestep),
        cross_gate_timestep=_sel(video_args.cross_gate_timestep),
    )


class LTXModelType(Enum):
    """Which modality stems and A↔V cross-attn tables this checkpoint owns."""

    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        """True when the video patch stem and video blocks exist."""
        return self in (LTXModelType.AudioVideo, LTXModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        """True when the audio patch stem and audio blocks exist."""
        return self in (LTXModelType.AudioVideo, LTXModelType.AudioOnly)


class LTXModel(torch.nn.Module):
    """Checkpoint-compatible LTX velocity DiT (video, audio, or joint AV)."""

    TRANSFORMER_BLOCK_CLS = BasicAVTransformerBlock

    def __init__(
        self,
        *,
        model_type: LTXModelType = LTXModelType.AudioVideo,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        norm_eps: float = 1e-06,
        ops: TransformerOpsConfig = DEFAULT_TRANSFORMER_OPS,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        timestep_scale_multiplier: int = 1000,
        use_middle_indices_grid: bool = True,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_in_channels: int = 128,
        audio_out_channels: int = 128,
        audio_cross_attention_dim: int = 2048,
        audio_positional_embedding_max_pos: list[int] | None = None,
        av_ca_timestep_scale_multiplier: int = 1,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        double_precision_rope: bool = False,
        apply_gated_attention: bool = False,
        caption_projection: torch.nn.Module | None = None,
        audio_caption_projection: torch.nn.Module | None = None,
        cross_attention_adaln: bool = False,
        use_prompt_adaln_single: bool = True,
        ff_bias: bool = True,
        audio_ff_bias: bool = True,
        use_keyframes_abs_pos_embedding: bool = False,
    ) -> None:
        """Build video and/or audio stems, AV cross-attn AdaLN, and the block stack."""
        super().__init__()
        self._enable_gradient_checkpointing = False
        self.cross_attention_adaln = cross_attention_adaln
        self.use_prompt_adaln_single = use_prompt_adaln_single
        self.use_keyframes_abs_pos_embedding = use_keyframes_abs_pos_embedding
        self.use_middle_indices_grid = use_middle_indices_grid
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.model_type = model_type
        cross_pe_max_pos = None
        if model_type.is_video_enabled():
            if positional_embedding_max_pos is None:
                positional_embedding_max_pos = [20, 2048, 2048]
            self.positional_embedding_max_pos = positional_embedding_max_pos
            self.num_attention_heads = num_attention_heads
            self.inner_dim = num_attention_heads * attention_head_dim
            self._init_video(
                in_channels=in_channels,
                out_channels=out_channels,
                norm_eps=norm_eps,
                caption_projection=caption_projection,
            )

        if model_type.is_audio_enabled():
            if audio_positional_embedding_max_pos is None:
                audio_positional_embedding_max_pos = [20]
            self.audio_positional_embedding_max_pos = audio_positional_embedding_max_pos
            self.audio_num_attention_heads = audio_num_attention_heads
            self.audio_inner_dim = self.audio_num_attention_heads * audio_attention_head_dim
            self._init_audio(
                in_channels=audio_in_channels,
                out_channels=audio_out_channels,
                norm_eps=norm_eps,
                caption_projection=audio_caption_projection,
            )

        if model_type.is_video_enabled() and model_type.is_audio_enabled():
            cross_pe_max_pos = max(self.positional_embedding_max_pos[0], self.audio_positional_embedding_max_pos[0])
            self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier
            self.audio_cross_attention_dim = audio_cross_attention_dim
            self._init_audio_video(num_scale_shift_values=4)

        self._init_preprocessors(cross_pe_max_pos)
        # Initialize transformer blocks
        self._init_transformer_blocks(
            num_layers=num_layers,
            attention_head_dim=attention_head_dim if model_type.is_video_enabled() else 0,
            cross_attention_dim=cross_attention_dim,
            audio_attention_head_dim=audio_attention_head_dim if model_type.is_audio_enabled() else 0,
            audio_cross_attention_dim=audio_cross_attention_dim,
            norm_eps=norm_eps,
            ops=ops,
            apply_gated_attention=apply_gated_attention,
            ff_bias=ff_bias,
            audio_ff_bias=audio_ff_bias,
        )
        # Hook for per-block input prep. Compile transforms in `compiling.py`
        # wrap (not replace) this with a processor that also marks the seq dim
        # dynamic, so any caller customisation here is preserved as the inner.
        self.block_input_processor = BlockPerturbationsProcessor()

    @property
    def _adaln_embedding_coefficient(self) -> int:
        return adaln_embedding_coefficient(self.cross_attention_adaln)

    @property
    def num_blocks(self) -> int:
        """Number of transformer blocks for mask preparation and block streaming."""
        return len(self.transformer_blocks)

    def __getattr__(self, name: str) -> Any:
        if name == "blocks" and "blocks" not in self._modules and "transformer_blocks" in self._modules:
            return self._modules["transformer_blocks"]
        return super().__getattr__(name)

    def configure_attention_backend(
        self,
        backend: str | None,
        *,
        self_only: bool = False,
        cross_only: bool = False,
        backend_options: Mapping[str, object] | None = None,
        dense_blocks: Iterable[int] = (),
        dense_backend: str | None = None,
    ) -> int:
        """Bind an attention backend to eligible LTX ``Attention`` modules.

        When ``self_only=True`` and ``backend="sol_attn"``, binds every self-attention
        module with ``dim_head == 128`` (e.g. LTX-2 video ``attn1``) while leaving
        cross-attention, 64-dim audio attention, and any ``dense_blocks`` indices on
        their exact dense backend (preserving e.g. ``flash_attention_3`` or defaulting
        to ``"torch"``).
        """
        if self_only and cross_only:
            raise ValueError("self_only and cross_only cannot both be True")

        from .attention import _APPROXIMATE_ATTENTION_BACKENDS, Attention

        normalized: str | None = None
        if backend is not None:
            from worldfoundry.core.attention.backends.probe import normalize_attention_backend

            normalized = normalize_attention_backend(str(backend))

        resolved_dense_backend: str | None = dense_backend
        if resolved_dense_backend is None and backend_options and "dense_backend" in backend_options:
            raw_opt_dense = backend_options["dense_backend"]
            resolved_dense_backend = None if raw_opt_dense is None else str(raw_opt_dense)
        if resolved_dense_backend is not None:
            from worldfoundry.core.attention.backends.probe import normalize_attention_backend

            resolved_dense_backend = normalize_attention_backend(resolved_dense_backend)

        num_blocks = len(self.transformer_blocks)
        dense_indices: set[int] = set()
        for raw_idx in dense_blocks:
            if isinstance(raw_idx, bool) or not isinstance(raw_idx, int):
                raise ValueError(f"dense_blocks entries must be integers, got {raw_idx!r}")
            idx = raw_idx + num_blocks if raw_idx < 0 else raw_idx
            if 0 <= idx < num_blocks:
                dense_indices.add(idx)

        dense_modules: set[int] = set()
        if dense_indices:
            for block_idx in dense_indices:
                for child in self.transformer_blocks[block_idx].modules():
                    if isinstance(child, Attention):
                        dense_modules.add(id(child))

        effective_options: Mapping[str, object] | None = backend_options
        if (
            resolved_dense_backend is not None
            and normalized in {"sol_attn", "cudnn_fp8"}
            and (backend_options is None or "dense_backend" not in backend_options)
        ):
            effective_options = {**(backend_options or {}), "dense_backend": resolved_dense_backend}

        configured = 0
        for child in self.modules():
            if not isinstance(child, Attention):
                continue
            if self_only and not child.is_self_attention:
                continue
            if cross_only and child.is_self_attention:
                continue
            if self_only and normalized == "sol_attn" and child.dim_head != 128:
                continue
            if id(child) in dense_modules:
                if resolved_dense_backend is not None:
                    child.set_attention_backend(resolved_dense_backend)
                elif normalized is None:
                    child.set_attention_backend(None)
                elif child.attention_backend in {None, "torch", "auto", *_APPROXIMATE_ATTENTION_BACKENDS}:
                    child.set_attention_backend(child._dense_attention_backend or "torch")
                # Otherwise preserve an existing exact dense backend (e.g. flash_attention_3).
                continue
            child.set_attention_backend(normalized, backend_options=effective_options)
            configured += 1
        if backend is not None and configured == 0:
            raise ValueError(
                f"no eligible LTX attention modules matched backend={backend!r} "
                f"(self_only={self_only}, cross_only={cross_only})"
            )
        return configured

    def fuse_projections(
        self,
        *,
        fuse_self_qkv: bool = True,
        fuse_cross_kv: bool = True,
    ) -> dict[str, int]:
        """Fuse LTX self-attention QKV and cross-attention KV projections in place."""
        from .fused_ops import fuse_ltx_projections

        return fuse_ltx_projections(self, fuse_self_qkv=fuse_self_qkv, fuse_cross_kv=fuse_cross_kv)

    def enable_fused_ops(self) -> int:
        """Install fused Triton QK-RMSNorm+RoPE, post-SA residual+RMSNorm, and gated FFN add."""
        from .fused_ops import enable_ltx_fused_ops

        return enable_ltx_fused_ops(self)

    def _keyframes_embedding(self) -> torch.Tensor | None:
        """Resolve the current parameter after checkpoint loading or device placement."""
        return getattr(self, "keyframes_abs_pos_embedding", None)

    def _init_video(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        caption_projection: torch.nn.Module | None = None,
    ) -> None:
        """Initialize video-specific components."""
        # Video input components
        self.patchify_proj = torch.nn.Linear(in_channels, self.inner_dim, bias=True)
        if caption_projection is not None:
            self.caption_projection = caption_projection

        self.keyframes_abs_pos_embedding = (
            torch.nn.Parameter(torch.zeros(1, self.inner_dim)) if self.use_keyframes_abs_pos_embedding else None
        )
        self.adaln_single = AdaLayerNormSingle(self.inner_dim, embedding_coefficient=self._adaln_embedding_coefficient)

        self.prompt_adaln_single = (
            AdaLayerNormSingle(self.inner_dim, embedding_coefficient=2)
            if self.cross_attention_adaln and self.use_prompt_adaln_single
            else None
        )

        # Video output components
        self.scale_shift_table = torch.nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = torch.nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
        self.proj_out = torch.nn.Linear(self.inner_dim, out_channels)

    def _init_audio(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        caption_projection: torch.nn.Module | None = None,
    ) -> None:
        """Initialize audio-specific components."""

        # Audio input components
        self.audio_patchify_proj = torch.nn.Linear(in_channels, self.audio_inner_dim, bias=True)
        if caption_projection is not None:
            self.audio_caption_projection = caption_projection

        self.audio_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=self._adaln_embedding_coefficient,
        )

        self.audio_prompt_adaln_single = (
            AdaLayerNormSingle(self.audio_inner_dim, embedding_coefficient=2)
            if self.cross_attention_adaln and self.use_prompt_adaln_single
            else None
        )

        # Audio output components
        self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(2, self.audio_inner_dim))
        self.audio_norm_out = torch.nn.LayerNorm(self.audio_inner_dim, elementwise_affine=False, eps=norm_eps)
        self.audio_proj_out = torch.nn.Linear(self.audio_inner_dim, out_channels)

    def _init_audio_video(
        self,
        num_scale_shift_values: int,
    ) -> None:
        """Initialize audio-video cross-attention components."""
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=1,
        )

        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=1,
        )

    def _init_preprocessors(
        self,
        cross_pe_max_pos: int | None = None,
    ) -> None:
        """Initialize preprocessors for LTX."""

        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            self.video_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                caption_projection=getattr(self, "caption_projection", None),
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
                action_embedder=getattr(self, "video_action_embedder", None),
                keyframes_embedding_provider=self._keyframes_embedding,
            )
            self.audio_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                caption_projection=getattr(self, "audio_caption_projection", None),
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )
        elif self.model_type.is_video_enabled():
            self.video_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                caption_projection=getattr(self, "caption_projection", None),
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
                keyframes_embedding_provider=self._keyframes_embedding,
            )
        elif self.model_type.is_audio_enabled():
            self.audio_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                caption_projection=getattr(self, "audio_caption_projection", None),
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )

    def refresh_preprocessor_bindings(self) -> None:
        """Rebind helpers after runtime policies replace registered submodules."""

        cross_pe_max_pos = None
        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            cross_pe_max_pos = max(
                self.positional_embedding_max_pos[0],
                self.audio_positional_embedding_max_pos[0],
            )
        self._init_preprocessors(cross_pe_max_pos)

    def _init_transformer_blocks(
        self,
        num_layers: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        norm_eps: float,
        ops: TransformerOpsConfig,
        apply_gated_attention: bool,
        ff_bias: bool = True,
        audio_ff_bias: bool = True,
    ) -> None:
        """Initialize transformer blocks for LTX."""
        video_config = (
            TransformerConfig(
                dim=self.inner_dim,
                heads=self.num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
                ff_bias=ff_bias,
            )
            if self.model_type.is_video_enabled()
            else None
        )
        audio_config = (
            TransformerConfig(
                dim=self.audio_inner_dim,
                heads=self.audio_num_attention_heads,
                d_head=audio_attention_head_dim,
                context_dim=audio_cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
                ff_bias=audio_ff_bias,
            )
            if self.model_type.is_audio_enabled()
            else None
        )
        self.transformer_blocks = torch.nn.ModuleList(
            [
                self.TRANSFORMER_BLOCK_CLS(
                    video=video_config,
                    audio=audio_config,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
                    ops=ops,
                )
                for _ in range(num_layers)
            ]
        )

    def set_gradient_checkpointing(self, enable: bool) -> None:
        """Enable or disable gradient checkpointing for transformer blocks.
        Gradient checkpointing trades compute for memory by recomputing activations
        during the backward pass instead of storing them. This can significantly
        reduce memory usage at the cost of ~20-30% slower training.
        Args:
            enable: Whether to enable gradient checkpointing
        """
        self._enable_gradient_checkpointing = enable

    def _keep_all_perturbations(self, args: TransformerArgs) -> BatchedPerturbationConfig:
        """All-keep masks for ``args``'s batch, in its token dtype and device."""
        return BatchedPerturbationConfig.empty(args.x.shape[0], self.num_blocks, args.x.device, args.x.dtype)

    def _needs_materialized_perturbations(self) -> bool:
        """True when a custom processor, block forward override, or legacy CudaGraphRunner requires explicit masks."""
        return (
            type(self.block_input_processor) is not BlockPerturbationsProcessor
            or hasattr(self._process_transformer_blocks, "_captures")
            or self.TRANSFORMER_BLOCK_CLS.forward is not BasicAVTransformerBlock.forward
        )

    def _process_transformer_blocks(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig | None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        """Process transformer blocks for LTXAV.
        Per-block perturbation masks are precomputed here and attached to each
        modality's ``TransformerArgs`` so the block forward has no per-block
        identity to specialise on — all blocks share a single Dynamo cache slot.
        When ``perturbations`` is ``None`` and the default processor and block
        forward are active, skips host-to-device mask materialization and
        redundant all-ones clones.
        """
        if (
            perturbations is None
            and type(self.block_input_processor) is BlockPerturbationsProcessor
            and self.TRANSFORMER_BLOCK_CLS.forward is BasicAVTransformerBlock.forward
        ):
            for block in self.transformer_blocks:
                video, audio = gradient_checkpoint_forward(
                    block,
                    self._enable_gradient_checkpointing and self.training,
                    False,
                    video,
                    audio,
                )
            return video, audio

        if perturbations is None:
            perturbations = self._keep_all_perturbations(video or audio)

        for block_idx, block in enumerate(self.transformer_blocks):
            if video is not None:
                video = self.block_input_processor(
                    video,
                    perturbations,
                    block_idx,
                    self_attn_type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
                    cross_attn_type=PerturbationType.SKIP_A2V_CROSS_ATTN,
                )
            if audio is not None:
                audio = self.block_input_processor(
                    audio,
                    perturbations,
                    block_idx,
                    self_attn_type=PerturbationType.SKIP_AUDIO_SELF_ATTN,
                    cross_attn_type=PerturbationType.SKIP_V2A_CROSS_ATTN,
                )

            video, audio = gradient_checkpoint_forward(
                block,
                self._enable_gradient_checkpointing and self.training,
                False,
                video,
                audio,
            )

        return video, audio

    def _run_blocks_maybe_graphed(
        self,
        video_args: TransformerArgs | None,
        audio_args: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig | None,
        piecewise_graph: Any,
        *,
        segment_name: str = "ltx_blocks",
    ) -> tuple[TransformerArgs | None, TransformerArgs | None, bool, str]:
        """Execute transformer blocks through ``piecewise_graph`` when eligible."""
        piecewise_seg_name = segment_name
        piecewise_eligible = (
            piecewise_graph is not None
            and perturbations is None
            and not self.training
            and not self._enable_gradient_checkpointing
            and type(self.block_input_processor) is BlockPerturbationsProcessor
            and not hasattr(self._process_transformer_blocks, "_captures")
            and (video_args is None or (video_args.video_control is None and video_args.attention_cache is None))
            and (audio_args is None or audio_args.attention_cache is None)
        )
        if not piecewise_eligible:
            video_out, audio_out = self._process_transformer_blocks(
                video=video_args,
                audio=audio_args,
                perturbations=perturbations,
            )
            return video_out, audio_out, False, piecewise_seg_name

        def _graphed_stack(
            v_in: TransformerArgs | None,
            a_in: TransformerArgs | None,
        ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
            v_out, a_out = self._process_transformer_blocks(video=v_in, audio=a_in, perturbations=None)
            return (
                None if v_out is None else v_out.x,
                None if a_out is None else a_out.x,
            )

        piecewise_graph.get_or_register_segment(
            piecewise_seg_name,
            _graphed_stack,
            extra_key=f"ltx_piecewise:{id(self)}:{piecewise_seg_name}",
        )
        vx_out, ax_out = piecewise_graph.run_segment(piecewise_seg_name, video_args, audio_args)
        video_out = None if video_args is None else replace(video_args, x=vx_out)
        audio_out = None if audio_args is None else replace(audio_args, x=ax_out)
        return video_out, audio_out, True, piecewise_seg_name

    def _process_output(
        self,
        scale_shift_table: torch.Tensor,
        norm_out: torch.nn.LayerNorm,
        proj_out: torch.nn.Linear,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Process output for LTXV."""
        # Apply scale-shift modulation
        scale_shift_values = (
            scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        x = norm_out(x)
        x = x * (1 + scale) + shift
        x = proj_out(x)
        return x

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig | None = None,
        *,
        feature_cache: Any = None,
        feature_cache_step: int = 0,
        feature_cache_total_steps: int | None = None,
        _worldfoundry_piecewise_graph: Any = None,
        _video_grid_shape: tuple[int, int, int] | None = None,
        _sol_force_dense: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Predict velocity for each enabled modality; unused streams return ``None``.

        Inputs are packed ``B T D`` latents inside :class:`Modality`.  STG
        perturbations skip selected self/cross-attn branches per sample;
        ``None`` keeps every branch, with masks in the token dtype.
        """
        if not self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if not self.model_type.is_audio_enabled() and audio is not None:
            raise ValueError("Audio is not enabled for this model")

        video_args = self.video_args_preprocessor.prepare(video, audio) if video is not None else None
        audio_args = self.audio_args_preprocessor.prepare(audio, video) if audio is not None else None
        # Materialize defaults only when a custom processor or legacy graph wrapper requires them.
        if perturbations is None and self._needs_materialized_perturbations():
            perturbations = self._keep_all_perturbations(video_args or audio_args)

        sol_schedule = getattr(self, "_worldfoundry_sol_schedule", None)
        orig_video_args = video_args
        tile_inv_perm: torch.Tensor | None = None
        tile_sink_tokens: int | None = None
        if (
            isinstance(sol_schedule, Mapping)
            and sol_schedule.get("tile_3d") is not None
            and not _sol_force_dense
            and video is not None
            and video_args is not None
            and video_args.enabled
            and video_args.x.numel() > 0
            and int(video_args.x.shape[1]) >= int(sol_schedule.get("min_tokens", 0))
            and not self.training
            and not torch.is_grad_enabled()
            and perturbations is None
            and video_args.self_attention_mask is None
            and video_args.video_control is None
            and video_args.attention_cache is None
        ):
            grid_shape = _video_grid_shape
            if grid_shape is None and not (
                video_args.x.device.type == "cuda" and torch.cuda.is_current_stream_capturing()
            ):
                grid_cache = self.__dict__.setdefault("_worldfoundry_video_grid_cache", {})
                grid_shape = resolve_cached_video_grid_shape(grid_cache, video.positions)
            if grid_shape is not None and grid_shape[0] * grid_shape[1] * grid_shape[2] == int(video_args.x.shape[1]):
                tf, th, tw = (int(v) for v in sol_schedule["tile_3d"])
                perm_cache: dict[Any, tuple[torch.Tensor, torch.Tensor, int] | None] = self.__dict__.setdefault(
                    "_worldfoundry_tile_perm_cache", {}
                )
                perm_key = (grid_shape[0], grid_shape[1], grid_shape[2], tf, th, tw, video_args.x.device)
                if perm_key in perm_cache:
                    cached_perm = perm_cache[perm_key]
                else:
                    cached_perm = _build_tile_3d_permutation(
                        grid_shape[0],
                        grid_shape[1],
                        grid_shape[2],
                        tf,
                        th,
                        tw,
                        video_args.x.device,
                    )
                    if len(perm_cache) >= 16:
                        perm_cache.pop(next(iter(perm_cache)))
                    perm_cache[perm_key] = cached_perm
                if cached_perm is not None:
                    tile_perm, tile_inv_perm, sink = cached_perm
                    video_args = _permute_video_args(video_args, tile_perm)
                    if sink > 0:
                        tile_sink_tokens = sink

        sol_attn_modules: list[Any] = []
        if _sol_force_dense or tile_sink_tokens is not None:
            for block in self.transformer_blocks:
                attn1 = getattr(block, "attn1", None)
                if attn1 is not None and getattr(attn1, "attention_backend", None) == "sol_attn":
                    sol_attn_modules.append(attn1)
            for mod in sol_attn_modules:
                if _sol_force_dense:
                    mod._worldfoundry_force_dense = True
                if tile_sink_tokens is not None:
                    mod._worldfoundry_sol_sink_tokens = tile_sink_tokens

        seg_label = "ltx_blocks"
        if _sol_force_dense:
            seg_label = f"{seg_label}_dense"
        elif tile_sink_tokens is not None:
            seg_label = f"{seg_label}_sink{tile_sink_tokens}"

        try:
            if feature_cache is None or perturbations is not None:
                video_out, audio_out, _, _ = self._run_blocks_maybe_graphed(
                    video_args,
                    audio_args,
                    perturbations,
                    _worldfoundry_piecewise_graph,
                    segment_name=seg_label,
                )
            else:
                if video_args is not None and audio_args is not None:
                    cache_signal = torch.cat(
                        (video_args.timesteps.mean(dim=1), audio_args.timesteps.mean(dim=1)),
                        dim=-1,
                    )
                elif video_args is not None:
                    cache_signal = video_args.timesteps
                else:
                    assert audio_args is not None
                    cache_signal = audio_args.timesteps

                ran_dense = [False]
                piecewise_used = [False]
                seg_name = [seg_label]

                def _compute_residual() -> torch.Tensor:
                    ran_dense[0] = True
                    v_out, a_out, used_pw, name = self._run_blocks_maybe_graphed(
                        video_args,
                        audio_args,
                        None,
                        _worldfoundry_piecewise_graph,
                        segment_name=seg_label,
                    )
                    piecewise_used[0] = used_pw
                    seg_name[0] = name
                    if video_args is not None and audio_args is not None:
                        assert v_out is not None and a_out is not None
                        return torch.cat(
                            ((v_out.x - video_args.x).reshape(-1), (a_out.x - audio_args.x).reshape(-1)),
                            dim=0,
                        )
                    if video_args is not None:
                        assert v_out is not None
                        return v_out.x - video_args.x
                    assert audio_args is not None and a_out is not None
                    return a_out.x - audio_args.x

                residual = feature_cache.run(
                    int(feature_cache_step),
                    cache_signal,
                    _compute_residual,
                    total_steps=feature_cache_total_steps,
                )
                if _worldfoundry_piecewise_graph is not None and not ran_dense[0]:
                    _worldfoundry_piecewise_graph.record_skip(seg_name[0])

                if video_args is not None and audio_args is not None:
                    v_numel = video_args.x.numel()
                    v_res = residual[:v_numel].view_as(video_args.x)
                    a_res = residual[v_numel:].view_as(audio_args.x)
                    video_out = replace(video_args, x=video_args.x + v_res)
                    audio_out = replace(audio_args, x=audio_args.x + a_res)
                elif video_args is not None:
                    video_out = replace(video_args, x=video_args.x + residual)
                    audio_out = None
                else:
                    assert audio_args is not None
                    video_out = None
                    audio_out = replace(audio_args, x=audio_args.x + residual)
        finally:
            for mod in sol_attn_modules:
                if _sol_force_dense:
                    mod._worldfoundry_force_dense = False
                if tile_sink_tokens is not None:
                    mod._worldfoundry_sol_sink_tokens = None

        if tile_inv_perm is not None and video_out is not None and orig_video_args is not None:
            video_out = replace(orig_video_args, x=video_out.x.index_select(1, tile_inv_perm))

        # Process output
        vx = (
            self._process_output(
                self.scale_shift_table, self.norm_out, self.proj_out, video_out.x, video_out.embedded_timestep
            )
            if video_out is not None
            else None
        )
        ax = (
            self._process_output(
                self.audio_scale_shift_table,
                self.audio_norm_out,
                self.audio_proj_out,
                audio_out.x,
                audio_out.embedded_timestep,
            )
            if audio_out is not None
            else None
        )
        return vx, ax


class LegacyX0Model(torch.nn.Module):
    """
    Legacy X0 model implementation.
    Returns fully denoised output based on the velocities produced by the base model.
    """

    def __init__(self, velocity_model: LTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    @property
    def num_blocks(self) -> int:
        """Delegate the block count to the velocity backbone."""
        return self.velocity_model.num_blocks

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
        sigma: float,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, audio, perturbations)
        denoised_video = to_denoised(video.latent, vx, sigma) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, sigma) if ax is not None else None
        return denoised_video, denoised_audio


class X0Model(LegacyX0Model):
    """Convert velocity to clean tokens using each modality's per-token timesteps."""

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, audio, perturbations)
        denoised_video = to_denoised(video.latent, vx, video.timesteps) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, audio.timesteps) if ax is not None else None
        return denoised_video, denoised_audio
