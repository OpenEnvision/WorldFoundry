"""Inference-only Qwen3-VL + MMDiT graph used by LDA-1B.

The module keeps the released checkpoint's module names and tensor shapes while
removing trainers, dataset readers, servers, and video-generation entrypoints.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from torch import Tensor, nn

from worldfoundry.base_models.llm_mllm_core.action.gr00t.modeling.embodiment_mlp import (
    SmallMLP as MLP,
)
from worldfoundry.core.attention import scaled_dot_product_attention

from .dinov3_configuration import DINOv3ViTConfig
from .dinov3_modeling import DINOv3ViTModel


def _namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(**{str(key): _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


class RMSNorm(nn.Module):
    """Parameter-compatible subset of the RMSNorm used by the released MMDiT."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        # The released state dict names this exact final-scale parameter ``g``.
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, value: Tensor) -> Tensor:
        return F.normalize(value, dim=-1) * self.g * self.scale


class Residual(nn.Module):
    """Single-stream residual connector; it intentionally has no state tensors."""

    def __init__(self, _streams: int = 1, *, dim: int | None = None) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, value: Tensor) -> tuple[Tensor, Any]:
        return value, lambda branch: value + branch


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps: Tensor) -> Tensor:
        dtype = next(self.parameters()).dtype
        return self.timestep_embedder(self.time_proj(timesteps).to(dtype))


class JointAttention(nn.Module):
    """Released two-stream projection layout backed by WorldFoundry exact SDPA."""

    def __init__(
        self,
        *,
        dim_inputs: tuple[int, ...],
        dim_head: int,
        heads: int,
        qk_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.num_inputs = len(dim_inputs)
        self.heads = heads
        self.dim_head = dim_head
        self.to_qkv = nn.ModuleList([nn.Linear(dim_input, inner_dim * 3, bias=False) for dim_input in dim_inputs])
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim_input, bias=False) for dim_input in dim_inputs])
        self.qk_rmsnorm = bool(qk_rmsnorm)
        if self.qk_rmsnorm:
            self.q_rmsnorms = nn.ModuleList([MultiHeadRMSNorm(dim_head, heads=heads) for _ in dim_inputs])
            self.k_rmsnorms = nn.ModuleList([MultiHeadRMSNorm(dim_head, heads=heads) for _ in dim_inputs])
        else:
            self.q_rmsnorms = (None,) * self.num_inputs
            self.k_rmsnorms = (None,) * self.num_inputs
        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    def forward(
        self,
        inputs: tuple[Tensor, ...],
        masks: tuple[Tensor | None, ...] | None = None,
    ) -> tuple[Tensor, ...]:
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} attention streams, got {len(inputs)}")
        masks = masks or (None,) * self.num_inputs
        projected: list[Tensor] = []
        keep_masks: list[Tensor] = []
        lengths: list[int] = []
        for value, mask, projection, q_norm, k_norm in zip(
            inputs, masks, self.to_qkv, self.q_rmsnorms, self.k_rmsnorms, strict=False
        ):
            batch, length, _ = value.shape
            lengths.append(length)
            qkv = projection(value).view(batch, length, 3, self.heads, self.dim_head)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            if self.qk_rmsnorm:
                query, key, val = qkv.unbind(0)
                qkv = torch.stack((q_norm(query), k_norm(key), val), dim=0)
            projected.append(qkv)
            if mask is None:
                mask = torch.ones(batch, length, dtype=torch.bool, device=value.device)
            else:
                mask = mask.to(device=value.device, dtype=torch.bool)
            keep_masks.append(mask)
        query, key, value = torch.cat(projected, dim=3).unbind(0)
        key_mask = torch.cat(keep_masks, dim=1)[:, None, None, :]
        output = scaled_dot_product_attention(query, key, value, attn_mask=key_mask)
        output = output.transpose(1, 2).reshape(output.shape[0], -1, self.heads * self.dim_head)
        chunks = output.split(lengths, dim=1)
        return tuple(projection(chunk) for chunk, projection in zip(chunks, self.to_out, strict=False))


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, *, heads: int = 1) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, value: Tensor) -> Tensor:
        return F.normalize(value, dim=-1) * self.gamma * self.scale


class MMDiTBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        final_dropout: bool = False,
        ff_inner_dim: int | None = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        qk_rmsnorm: bool = False,
        num_residual_streams: int = 1,
        **_: Any,
    ) -> None:
        super().__init__()
        if num_residual_streams != 1:
            raise ValueError("LDA-1B inference supports the released single residual stream only")
        self.image_attn_residual_fn = Residual(1, dim=dim)
        self.image_cross_attn_residual_fn = Residual(1, dim=dim)
        self.image_ff_residual_fn = Residual(1, dim=dim)
        self.action_attn_residual_fn = Residual(1, dim=dim)
        self.action_cross_attn_residual_fn = Residual(1, dim=dim)
        self.action_ff_residual_fn = Residual(1, dim=dim)

        gamma_dims = (dim,) * 8
        beta_dims = (dim,) * 4
        self.cond_dims = (*gamma_dims, *beta_dims)
        cond_linear = nn.Linear(dim, sum(self.cond_dims))
        self.to_cond = nn.Sequential(nn.Unflatten(1, (1, dim)), nn.SiLU(), cond_linear)
        nn.init.zeros_(cond_linear.weight)
        nn.init.zeros_(cond_linear.bias)
        nn.init.constant_(cond_linear.bias[: sum(gamma_dims)], 1.0)

        self.image_attn_layernorm = nn.LayerNorm(dim, elementwise_affine=False)
        self.action_attn_layernorm = nn.LayerNorm(dim, elementwise_affine=False)
        self.image_cross_attn_layernorm = nn.LayerNorm(dim, elementwise_affine=False)
        self.action_cross_attn_layernorm = nn.LayerNorm(dim, elementwise_affine=False)
        self.image_ff_layernorm = nn.LayerNorm(dim, elementwise_affine=False)
        self.action_ff_layernorm = nn.LayerNorm(dim, elementwise_affine=False)
        self.img_cross_attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )
        self.action_cross_attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )
        self.joint_attn = JointAttention(
            dim_inputs=(dim, dim),
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_rmsnorm=qk_rmsnorm,
        )
        self.image_ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        self.action_ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        *,
        text_tokens: Tensor,
        image_tokens: Tensor,
        action_tokens: Tensor,
        text_mask: Tensor | None,
        time_cond: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        (
            image_pre_attn_gamma,
            image_post_attn_gamma,
            image_pre_ff_gamma,
            image_post_ff_gamma,
            action_pre_attn_gamma,
            action_post_attn_gamma,
            action_pre_ff_gamma,
            action_post_ff_gamma,
            image_pre_attn_beta,
            image_pre_ff_beta,
            action_pre_attn_beta,
            action_pre_ff_beta,
        ) = self.to_cond(time_cond).split(self.cond_dims, dim=-1)
        image_tokens, add_image = self.image_attn_residual_fn(image_tokens)
        action_tokens, add_action = self.action_attn_residual_fn(action_tokens)
        image_tokens = self.image_attn_layernorm(image_tokens) * image_pre_attn_gamma + image_pre_attn_beta
        action_tokens = self.action_attn_layernorm(action_tokens) * action_pre_attn_gamma + action_pre_attn_beta
        image_tokens, action_tokens = self.joint_attn((image_tokens, action_tokens))
        image_tokens = add_image(image_tokens)
        action_tokens = add_action(action_tokens)

        image_tokens, add_image = self.image_cross_attn_residual_fn(image_tokens)
        action_tokens, add_action = self.action_cross_attn_residual_fn(action_tokens)
        image_tokens = self.img_cross_attn(
            self.image_cross_attn_layernorm(image_tokens),
            encoder_hidden_states=text_tokens,
            attention_mask=text_mask,
        )
        action_tokens = self.action_cross_attn(
            self.action_cross_attn_layernorm(action_tokens),
            encoder_hidden_states=text_tokens,
            attention_mask=text_mask,
        )
        image_tokens = add_image(image_tokens) * image_post_attn_gamma
        action_tokens = add_action(action_tokens) * action_post_attn_gamma

        image_tokens, add_image = self.image_ff_residual_fn(image_tokens)
        action_tokens, add_action = self.action_ff_residual_fn(action_tokens)
        image_tokens = self.image_ff(self.image_ff_layernorm(image_tokens) * image_pre_ff_gamma + image_pre_ff_beta)
        action_tokens = self.action_ff(
            self.action_ff_layernorm(action_tokens) * action_pre_ff_gamma + action_pre_ff_beta
        )
        image_tokens = add_image(image_tokens * image_post_ff_gamma)
        action_tokens = add_action(action_tokens * action_post_ff_gamma)
        return text_tokens, image_tokens, action_tokens


class MMDiT(nn.Module):
    def __init__(self, **config: Any) -> None:
        super().__init__()
        resolved = {
            "dropout": 0.1,
            "attention_bias": True,
            "activation_fn": "gelu-approximate",
            "upcast_attention": False,
            "norm_elementwise_affine": False,
            "norm_eps": 1.0e-5,
            "final_dropout": True,
            "positional_embeddings": "sinusoidal",
            "final_norm": True,
            "num_residual_streams": 1,
            **config,
        }
        self.config = _namespace(resolved)
        streams = int(resolved["num_residual_streams"])
        if streams != 1:
            raise ValueError("LDA-1B checkpoint requires num_residual_streams=1")
        inner_dim = int(resolved["num_attention_heads"]) * int(resolved["attention_head_dim"])
        self.inner_dim = inner_dim
        self.timestep_encoder = TimestepEncoder(inner_dim)
        self.text_attn_layernorm = nn.LayerNorm(int(resolved["cross_attention_dim"]), elementwise_affine=False)
        block_kwargs = dict(resolved)
        block_kwargs.pop("num_layers", None)
        block_kwargs.pop("output_dim", None)
        block_kwargs.pop("num_residual_streams", None)
        block_kwargs.pop("input_embedding_dim", None)
        block_kwargs.pop("norm_type", None)
        block_kwargs.pop("interleave_self_attention", None)
        block_kwargs.pop("positional_embeddings", None)
        block_kwargs.pop("final_norm", None)
        self.blocks = nn.ModuleList(
            [
                MMDiTBlock(dim=inner_dim, num_residual_streams=streams, **block_kwargs)
                for _ in range(int(resolved["num_layers"]))
            ]
        )
        self.norm = RMSNorm(inner_dim) if bool(resolved["final_norm"]) else nn.Identity()
        self.action_norm = RMSNorm(inner_dim) if bool(resolved["final_norm"]) else nn.Identity()
        self.action_proj_out = nn.Linear(inner_dim, int(resolved["output_dim"]))
        self.image_proj_out = nn.Linear(inner_dim, int(resolved["output_dim"]))

    def forward(
        self,
        *,
        image_tokens: Tensor,
        action_tokens: Tensor,
        text_tokens: Tensor,
        register_tokens: Tensor | None,
        text_mask: Tensor | None,
        time_cond: Tensor,
        task_embedding: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        register_count = 0 if register_tokens is None else register_tokens.shape[1]
        if register_tokens is not None:
            image_tokens = torch.cat((register_tokens, image_tokens), dim=1)
        text_tokens = self.text_attn_layernorm(text_tokens)
        condition = self.timestep_encoder(time_cond)
        if task_embedding is not None:
            condition = condition + task_embedding
        for block in self.blocks:
            text_tokens, image_tokens, action_tokens = block(
                text_tokens=text_tokens,
                image_tokens=image_tokens,
                action_tokens=action_tokens,
                text_mask=text_mask,
                time_cond=condition,
            )
        if register_count:
            image_tokens = image_tokens[:, register_count:]
        return (
            self.image_proj_out(self.norm(image_tokens)),
            self.action_proj_out(self.action_norm(action_tokens)),
        )


def swish(value: Tensor) -> Tensor:
    return value * torch.sigmoid(value)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: Tensor) -> Tensor:
        half = self.embedding_dim // 2
        exponent = -torch.arange(half, dtype=torch.float32, device=timesteps.device) * (
            torch.log(torch.tensor(10000.0, device=timesteps.device)) / half
        )
        frequencies = timesteps.float().unsqueeze(-1) * exponent.exp()
        return torch.cat((torch.sin(frequencies), torch.cos(frequencies)), dim=-1)


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_categories = num_categories
        self.W = nn.Parameter(torch.empty(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.empty(num_categories, hidden_dim))

    def forward(self, value: Tensor, category_ids: Tensor) -> Tensor:
        return torch.bmm(value, self.W[category_ids]) + self.b[category_ids].unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, value: Tensor, category_ids: Tensor) -> Tensor:
        return self.layer2(F.relu(self.layer1(value, category_ids)), category_ids)


class ActionEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: Tensor, timesteps: Tensor) -> Tensor:
        batch, horizon, _ = actions.shape
        if timesteps.ndim != 1 or timesteps.shape[0] != batch:
            raise ValueError("timesteps must have shape (batch,)")
        timesteps = timesteps.unsqueeze(1).expand(-1, horizon)
        action_embedding = self.layer1(actions)
        time_embedding = self.pos_encoding(timesteps).to(action_embedding.dtype)
        return self.layer3(swish(self.layer2(torch.cat((action_embedding, time_embedding), dim=-1))))


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int, num_embodiments: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: Tensor, timesteps: Tensor, category_ids: Tensor) -> Tensor:
        batch, horizon, _ = actions.shape
        if timesteps.ndim != 1 or timesteps.shape[0] != batch:
            raise ValueError("timesteps must have shape (batch,)")
        timesteps = timesteps.unsqueeze(1).expand(-1, horizon)
        action_embedding = self.W1(actions, category_ids)
        time_embedding = self.pos_encoding(timesteps).to(action_embedding.dtype)
        return self.W3(
            swish(self.W2(torch.cat((action_embedding, time_embedding), dim=-1), category_ids)),
            category_ids,
        )


class FlowmatchingActionHead(nn.Module):
    """Released policy sampler with the training and video branches removed."""

    def __init__(self, config: Mapping[str, Any], vision_config: DINOv3ViTConfig) -> None:
        super().__init__()
        cfg = _namespace(config)
        self.config = cfg
        action_model_shapes = {
            "DiT-B": (768, 64, 12),
            "DiT-L": (1536, 48, 32),
            "DiT-XL": (2048, 64, 32),
        }
        try:
            input_dim, head_dim, heads = action_model_shapes[str(cfg.action_model_type)]
        except KeyError as exc:
            raise ValueError(f"Unsupported LDA-1B action_model_type {cfg.action_model_type!r}") from exc
        diffusion = dict(config["diffusion_model_cfg"])
        diffusion.update(
            input_embedding_dim=input_dim,
            attention_head_dim=head_dim,
            num_attention_heads=heads,
        )
        self.model = MMDiT(**diffusion)
        self.hidden_size = int(cfg.hidden_size)
        self.input_embedding_dim = input_dim
        self.action_dim = int(cfg.action_dim)
        self.state_dim = int(cfg.state_dim) if cfg.state_dim is not None else None
        self.action_horizon = int(cfg.future_action_window_size) + 1
        self.num_inference_timesteps = int(cfg.num_inference_timesteps)
        self.vision_encoder_type = str(cfg.vision_encoder_type)
        if self.vision_encoder_type != "dinov3":
            raise ValueError("The released LDA-1B checkpoints require vision_encoder_type=dinov3")
        self.num_views = int(cfg.num_views)
        self.obs_horizon = int(cfg.obs_horizon)
        self.multi_embodiment = int(cfg.max_num_embodiments) > 1
        if self.multi_embodiment:
            self.state_encoder = (
                CategorySpecificMLP(int(cfg.max_num_embodiments), self.state_dim, self.hidden_size, input_dim)
                if self.state_dim is not None
                else None
            )
            self.action_encoder = MultiEmbodimentActionEncoder(self.action_dim, input_dim, int(cfg.max_num_embodiments))
            self.action_decoder = CategorySpecificMLP(
                int(cfg.max_num_embodiments),
                int(diffusion["output_dim"]),
                self.hidden_size,
                self.action_dim,
            )
        else:
            self.state_encoder = (
                MLP(self.state_dim, self.hidden_size, input_dim) if self.state_dim is not None else None
            )
            self.action_encoder = ActionEncoder(self.action_dim, input_dim)
            self.action_decoder = MLP(int(diffusion["output_dim"]), self.hidden_size, self.action_dim)
        self.vision_encoder = DINOv3ViTModel(vision_config)
        self.img_size = vision_config.image_size
        num_channels = int(vision_config.hidden_size)
        grid = int(vision_config.image_size) // int(vision_config.patch_size)
        self.orig_patch_shape = (grid, grid)
        total_obs_len = self.obs_horizon + 1
        self.obs_merger = nn.Linear(num_channels * total_obs_len, input_dim)
        self.obs_projector = nn.Linear(self.hidden_size, num_channels)
        self.obs_len = grid * grid * self.num_views
        self.register_tokens = nn.Embedding(int(cfg.num_target_vision_tokens), input_dim)
        self.next_obs_learnable_tokens = nn.Parameter(0.02 * torch.randn(num_channels))
        self.action_learnable_tokens = nn.Embedding(self.action_horizon, input_dim)
        inner_dim = heads * head_dim
        self.policy_embedding = nn.Parameter(0.02 * torch.randn(inner_dim))
        self.fd_embedding = nn.Parameter(0.02 * torch.randn(inner_dim))
        self.vg_embedding = nn.Parameter(0.02 * torch.randn(inner_dim))
        self.id_embedding = nn.Parameter(0.02 * torch.randn(inner_dim))
        if bool(cfg.add_pos_embed):
            self.position_embedding = nn.Embedding(int(cfg.max_seq_len), input_dim)
        self.num_timestep_buckets = int(cfg.num_timestep_buckets)

    def encode_observations(self, pixels: Tensor) -> Tensor:
        batch, views, timesteps = pixels.shape[:3]
        flat = pixels.reshape(batch * views * timesteps, *pixels.shape[3:])
        output = self.vision_encoder(flat).last_hidden_state
        output = output.reshape(batch, views, timesteps, output.shape[1], output.shape[2])
        return output.permute(0, 1, 3, 4, 2).reshape(batch, views * output.shape[3], -1)

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: Tensor,
        *,
        state: Tensor | None,
        curr_pixels: Tensor,
        embodiment_id: Tensor,
        attention_mask: Tensor | None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        batch = vl_embs.shape[0]
        curr_obs = self.encode_observations(curr_pixels)
        actions = torch.randn(
            (batch, self.action_horizon, self.action_dim),
            dtype=vl_embs.dtype,
            device=vl_embs.device,
            generator=generator,
        )
        if self.state_dim is not None and state is None:
            raise ValueError("LDA-1B requires state input")
        if self.multi_embodiment:
            state_features = self.state_encoder(state, embodiment_id) if state is not None else None
        else:
            state_features = self.state_encoder(state) if state is not None else None
        task_embedding = self.policy_embedding.unsqueeze(0).expand(batch, -1)
        steps = self.num_inference_timesteps
        if steps <= 0:
            raise ValueError("num_inference_timesteps must be positive")
        for step in range(steps):
            timestep_value = int((step / float(steps)) * self.num_timestep_buckets)
            timesteps = torch.full((batch,), timestep_value, device=vl_embs.device, dtype=torch.long)
            if self.multi_embodiment:
                action_features = self.action_encoder(actions, timesteps, embodiment_id)
            else:
                action_features = self.action_encoder(actions, timesteps)
            noisy_next_obs = self.next_obs_learnable_tokens[None, None].expand(batch, curr_obs.shape[1], -1)
            obs_tokens = self.obs_merger(torch.cat((curr_obs, noisy_next_obs), dim=-1))
            register_tokens = self.register_tokens.weight.unsqueeze(0).expand(batch, -1, -1)
            if state_features is not None:
                action_features = torch.cat((state_features, action_features), dim=1)
            if bool(self.config.add_pos_embed):
                obs_positions = torch.arange(obs_tokens.shape[1], device=vl_embs.device)
                obs_tokens = obs_tokens + self.position_embedding(obs_positions).unsqueeze(0)
                action_positions = torch.arange(
                    obs_tokens.shape[1],
                    obs_tokens.shape[1] + action_features.shape[1],
                    device=vl_embs.device,
                )
                action_features = action_features + self.position_embedding(action_positions).unsqueeze(0)
            _, action_tokens = self.model(
                image_tokens=obs_tokens,
                action_tokens=action_features,
                text_tokens=vl_embs,
                register_tokens=register_tokens,
                text_mask=attention_mask,
                time_cond=timesteps,
                task_embedding=task_embedding,
            )
            if self.multi_embodiment:
                decoded = self.action_decoder(action_tokens, embodiment_id)
            else:
                decoded = self.action_decoder(action_tokens)
            actions = actions + decoded[:, -self.action_horizon :] / float(steps)
        return actions


class QwenVLInterface(nn.Module):
    def __init__(self, model_config: Any) -> None:
        super().__init__()
        from transformers import Qwen3VLForConditionalGeneration

        self.model = Qwen3VLForConditionalGeneration(model_config)
        self.model.config.hidden_size = self.model.config.text_config.hidden_size


class LDAInferenceModel(nn.Module):
    """Checkpoint-key-compatible top-level inference assembly."""

    def __init__(
        self,
        *,
        qwen_config: Any,
        action_config: Mapping[str, Any],
        vision_config: DINOv3ViTConfig,
    ) -> None:
        super().__init__()
        hidden_size = int(qwen_config.text_config.hidden_size)
        expected = int(action_config["diffusion_model_cfg"]["cross_attention_dim"])
        if hidden_size != expected:
            raise ValueError(f"Qwen hidden size {hidden_size} does not match action cross-attention {expected}")
        self.qwen_vl_interface = QwenVLInterface(qwen_config)
        self.action_model = FlowmatchingActionHead(action_config, vision_config)


__all__ = [
    "ActionEncoder",
    "FlowmatchingActionHead",
    "JointAttention",
    "LDAInferenceModel",
    "MMDiT",
    "MultiEmbodimentActionEncoder",
]
