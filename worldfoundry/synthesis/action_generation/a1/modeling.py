"""Native AffordVLA/Molmo model and action heads used by A1 inference."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from worldfoundry.core.attention import scaled_dot_product_attention

from .configuration import A1Config


class A1ModelOutput(NamedTuple):
    last_hidden_state: torch.Tensor
    attn_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None


class ProprioProjector(nn.Module):
    """Checkpoint-compatible two-layer state projector."""

    def __init__(self, llm_dim: int, proprio_dim: int, device: Any = None) -> None:
        super().__init__()
        self.fc1 = nn.Linear(proprio_dim, llm_dim, bias=True, device=device)
        self.fc2 = nn.Linear(llm_dim, llm_dim, bias=True, device=device)
        self.act_fn1 = nn.GELU()

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_fn1(self.fc1(proprio)))


class MLPResNetBlock(nn.Module):
    def __init__(self, dim: int, device: Any = None) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim, device=device),
            nn.Linear(dim, dim, device=device),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(x)


class MLPResNet(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        device: Any = None,
    ) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim, device=device)
        self.fc1 = nn.Linear(input_dim, hidden_dim, device=device)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList([MLPResNetBlock(hidden_dim, device=device) for _ in range(num_blocks)])
        self.layer_norm2 = nn.LayerNorm(hidden_dim, device=device)
        self.fc2 = nn.Linear(hidden_dim, output_dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(self.layer_norm1(x)))
        for block in self.mlp_resnet_blocks:
            x = block(x)
        return self.fc2(self.layer_norm2(x))


class L1RegressionActionHead(nn.Module):
    """Continuous action head used by the public task checkpoints."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
        action_token_dim: int,
        num_actions_chunk: int,
        device: Any = None,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.action_token_dim = action_token_dim
        self.num_actions_chunk = num_actions_chunk
        self.model = MLPResNet(
            num_blocks=2,
            input_dim=input_dim * action_token_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            device=device,
        )

    def predict_action(self, hidden_states: torch.Tensor) -> torch.Tensor:
        expected = self.num_actions_chunk * self.action_token_dim
        if hidden_states.shape[1] != expected:
            raise ValueError(f"A1 action head expected {expected} hidden tokens, got {hidden_states.shape[1]}")
        reshaped = hidden_states.reshape(hidden_states.shape[0], self.num_actions_chunk, -1)
        return self.model(reshaped)

    forward = predict_action


class RmsNorm(nn.Module):
    """Timm-compatible RMSNorm without a timm runtime dependency."""

    def __init__(self, dim: int, eps: float = 1e-6, device: Any = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=device))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        source_dtype = x.dtype
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x.float() * torch.rsqrt(variance + self.eps)).to(source_dtype) * self.weight


class SelfAttention(nn.Module):
    """Parameter-compatible self attention for the released DiT blocks."""

    def __init__(self, dim: int, num_heads: int, device: Any = None) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("DiT hidden size must be divisible by its head count")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True, device=device)
        self.q_norm = RmsNorm(self.head_dim, device=device)
        self.k_norm = RmsNorm(self.head_dim, device=device)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim, device=device)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = x.transpose(1, 2).reshape(batch, length, dim)
        return self.proj_drop(self.proj(x))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, device: Any = None) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("DiT hidden size must be divisible by its head count")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=True, device=device)
        self.kv = nn.Linear(dim, dim * 2, bias=True, device=device)
        self.q_norm = RmsNorm(self.head_dim, device=device)
        self.k_norm = RmsNorm(self.head_dim, device=device)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim, device=device)
        self.proj_drop = nn.Dropout(0.0)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, target_len, dim = x.shape
        source_len = condition.shape[1]
        q = self.q(x).reshape(batch, target_len, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(condition).reshape(batch, source_len, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        attn_mask = None
        if mask is not None:
            attn_mask = mask.to(torch.bool).reshape(batch, 1, 1, source_len)
        x = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=self.scale,
        )
        x = x.transpose(1, 2).reshape(batch, target_len, dim)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    """Timm MLP parameter layout used by DiT."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int | None = None,
        device: Any = None,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, device=device)
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(0.0)
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features or in_features, device=device)
        self.drop2 = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.norm(self.drop1(self.act(self.fc1(x))))))


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, device: Any = None) -> None:
        super().__init__()
        self.norm1 = RmsNorm(hidden_size, device=device)
        self.attn = SelfAttention(hidden_size, num_heads, device=device)
        self.cross_attn = CrossAttention(hidden_size, num_heads, device=device)
        self.norm2 = RmsNorm(hidden_size, device=device)
        self.ffn = Mlp(hidden_size, hidden_size, device=device)
        self.norm3 = RmsNorm(hidden_size, device=device)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), condition, mask)
        return x + self.ffn(self.norm3(x))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_dim: int, device: Any = None) -> None:
        super().__init__()
        self.norm_final = RmsNorm(hidden_size, device=device)
        self.ffn_final = Mlp(hidden_size, hidden_size, output_dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn_final(self.norm_final(x))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, device: Any = None) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, device=device),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, device=device),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / max(half, 1)
        )
        args = t.reshape(-1, 1).float() * frequencies.reshape(1, -1)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        dtype = self.mlp[0].weight.dtype
        return self.mlp(self.timestep_embedding(timestep, self.frequency_embedding_size).to(dtype))


def _sincos_1d(embed_dim: int, length: int) -> torch.Tensor:
    if embed_dim % 2:
        raise ValueError("Sin/cos embedding dimension must be even")
    positions = torch.arange(length, dtype=torch.float64)
    omega = torch.arange(embed_dim // 2, dtype=torch.float64) / (embed_dim / 2.0)
    omega = 1.0 / (10000.0**omega)
    angles = torch.outer(positions, omega)
    return torch.cat([angles.sin(), angles.cos()], dim=1).float()


def _multimodal_positions(embed_dim: int, lengths: OrderedDict[str, int]) -> torch.Tensor:
    modality = _sincos_1d(embed_dim // 2, len(lengths))
    output = []
    for index, length in enumerate(lengths.values()):
        item = torch.zeros(abs(length), embed_dim)
        item[:, : embed_dim // 2] = modality[index]
        item[:, embed_dim // 2 :] = _sincos_1d(embed_dim // 2, abs(length))
        output.append(item)
    return torch.cat(output, dim=0)


class DiT(nn.Module):
    """Diffusion transformer conditioned on the action-token Molmo states."""

    def __init__(
        self,
        output_dim: int,
        horizon: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        llm_state_cond_len: int,
        llm_state_cond_dim: int,
        device: Any = None,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.t_embedder = TimestepEmbedder(hidden_size, device=device)
        x_position = _multimodal_positions(hidden_size, OrderedDict((("timestep", 1), ("action", horizon)))).to(
            device=device
        )
        self.x_pos_embed = nn.Parameter(x_position.unsqueeze(0))
        self.llm_state_cond_pos_embed = nn.Parameter(
            _sincos_1d(hidden_size, llm_state_cond_len).to(device=device).unsqueeze(0)
        )
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, device=device) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, output_dim, device=device)
        self.llm_state_cond_dim = llm_state_cond_dim

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        llm_state: torch.Tensor,
        llm_state_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        time = self.t_embedder(timestep).unsqueeze(1)
        if time.shape[0] == 1 and x.shape[0] != 1:
            time = time.expand(x.shape[0], -1, -1)
        x = torch.cat([time, x], dim=1) + self.x_pos_embed[:, : x.shape[1] + 1].to(x.dtype)
        llm_state = llm_state + self.llm_state_cond_pos_embed[:, : llm_state.shape[1]].to(llm_state.dtype)
        for block in self.blocks:
            x = block(x, llm_state, llm_state_mask)
        return self.final_layer(x)[:, -self.horizon :]


class DiffusionTransformerActionHead(nn.Module):
    def __init__(self, config: A1Config, device: Any = None) -> None:
        super().__init__()
        self.action_dim = config.fixed_action_dim
        self.action_horizon = config.num_actions_chunk
        self.num_diffusion_steps = config.num_diffusion_steps
        self.num_diffusion_inference_steps = config.num_diffusion_inference_steps
        hidden = config.action_head_dit_hidden_size
        self.model = DiT(
            output_dim=self.action_dim,
            horizon=self.action_horizon,
            hidden_size=hidden,
            depth=config.action_head_dit_depth,
            num_heads=config.action_head_dit_num_heads,
            llm_state_cond_len=config.action_token_dim * config.num_actions_chunk,
            llm_state_cond_dim=config.d_model,
            device=device,
        )
        self.action_adaptor = nn.Sequential(
            nn.Linear(self.action_dim, hidden, device=device),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, hidden, device=device),
        )
        self.condition_adaptor = nn.Sequential(
            nn.Linear(config.d_model, hidden, device=device),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, hidden, device=device),
        )

    def predict_noise_or_sample(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        dtype = self.model.t_embedder.mlp[0].weight.dtype
        return self.model(
            self.action_adaptor(noisy_action.to(dtype)),
            timestep,
            self.condition_adaptor(hidden_states.to(dtype)),
        ).to(hidden_states.dtype)

    @torch.no_grad()
    def condition_sampling(
        self,
        hidden_states: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        try:
            from diffusers import DPMSolverMultistepScheduler
        except ImportError as error:
            raise RuntimeError(
                "A1 diffusion checkpoints require the diffusers package for the released DPM solver"
            ) from error
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=self.num_diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="sample",
        )
        scheduler.set_timesteps(self.num_diffusion_inference_steps, device=hidden_states.device)
        action = torch.randn(
            hidden_states.shape[0],
            self.action_horizon,
            self.action_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
            generator=generator,
        )
        for timestep in scheduler.timesteps:
            model_output = self.predict_noise_or_sample(
                action,
                timestep.reshape(1).expand(hidden_states.shape[0]),
                hidden_states,
            )
            action = scheduler.step(model_output, timestep, action).prev_sample.to(hidden_states.dtype)
        return action


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return TimestepEmbedder.timestep_embedding(x, self.dim)


class FlowMatchingActionHead(nn.Module):
    """Released Qwen2 action expert used by flow-matching variants."""

    def __init__(self, config: A1Config, device: Any = None) -> None:
        super().__init__()
        try:
            from transformers import Qwen2Config, Qwen2ForCausalLM
        except ImportError as error:
            raise RuntimeError("A1 flow-matching checkpoints require transformers") from error
        self.action_dim = config.fixed_action_dim
        self.proprio_dim = config.proprio_dim
        self.horizon = config.num_actions_chunk
        self.qwen2_hidden = config.action_head_flow_matching_dim
        self.qwen2_num_layers = config.action_head_flow_matching_layers
        self.pvf_func = config.action_head_flow_matching_pvf_function
        self.time_encoder = SinusoidalPositionalEncoding(self.qwen2_hidden)
        self.state_proj = nn.Linear(self.proprio_dim, self.qwen2_hidden, device=device)
        self.action_in_proj = nn.Linear(self.action_dim, self.qwen2_hidden, device=device)
        self.action_time_in = nn.Linear(self.qwen2_hidden * 2, self.qwen2_hidden, device=device)
        self.action_time_out = nn.Linear(self.qwen2_hidden, self.qwen2_hidden, device=device)
        qwen_config = Qwen2Config(
            hidden_size=self.qwen2_hidden,
            num_hidden_layers=self.qwen2_num_layers,
            num_attention_heads=config.action_head_flow_matching_heads,
            intermediate_size=config.action_head_flow_matching_intermediate_size,
            num_key_value_heads=config.action_head_flow_matching_kv_heads,
        )
        self.qwen2 = Qwen2ForCausalLM(qwen_config)
        if hasattr(self.qwen2.model, "embed_tokens"):
            self.qwen2.model.embed_tokens = None
        self.action_out = MLPResNet(
            2,
            self.qwen2_hidden,
            self.qwen2_hidden,
            self.action_dim,
            device=device,
        )

    def build_suffix_tokens(
        self,
        state: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        dtype = self.state_proj.weight.dtype
        state = state.reshape(state.shape[0], -1, state.shape[-1])[:, 0]
        state_token = self.state_proj(state.to(dtype)).unsqueeze(1)
        time_token = self.time_encoder(timestep).to(dtype).unsqueeze(1).expand(-1, x_t.shape[1], -1)
        action_token = self.action_in_proj(x_t.to(dtype))
        action_time = self.action_time_out(F.silu(self.action_time_in(torch.cat([action_token, time_token], -1))))
        return torch.cat([state_token, action_time], dim=1)

    def predict_vector_field(
        self,
        past_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]],
        state: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        *,
        valid_prefix_lengths: torch.Tensor,
    ) -> torch.Tensor:
        from transformers.cache_utils import DynamicCache

        target = self.build_suffix_tokens(state, x_t, timestep).to(next(self.qwen2.parameters()).dtype)
        past_len = int(past_key_values[0][0].shape[-2])
        prefix_positions = torch.arange(past_len, device=target.device).unsqueeze(0)
        prefix_mask = prefix_positions < valid_prefix_lengths.to(target.device).reshape(-1, 1)
        suffix_mask = torch.ones(target.shape[:2], dtype=torch.bool, device=target.device)
        attention_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
        legacy_cache = tuple(past_key_values)
        # Transformers 5 removed ``from_legacy_cache`` and made the legacy
        # key/value iterable the first DynamicCache constructor argument.
        # Retain the older path as well so the in-tree runtime works across
        # both supported cache APIs without pinning a Transformers downgrade.
        from_legacy_cache = getattr(DynamicCache, "from_legacy_cache", None)
        cache = from_legacy_cache(legacy_cache) if from_legacy_cache is not None else DynamicCache(legacy_cache)
        output = self.qwen2.model(
            inputs_embeds=target,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=False,
        )
        return self.action_out(output.last_hidden_state[:, -self.horizon :]).to(x_t.dtype)


class AffordVLA(nn.Module):
    """Inference graph joining Molmo vision/language features with an action head."""

    def __init__(self, config: A1Config, device: Any = None) -> None:
        super().__init__()
        from worldfoundry.synthesis.action_generation.molmobot.modeling.llm import Llm
        from worldfoundry.synthesis.action_generation.molmobot.modeling.vision_backbone import (
            MolmoVisionBackbone,
        )
        from worldfoundry.synthesis.action_generation.molmobot.torch_utils import BufferCache

        self.config = config
        self._cache = BufferCache()
        llm_config = config.build_llm_config(device=str(device) if device is not None else None)
        self.transformer = Llm(llm_config, self._cache, device)
        self.vision_backbone = MolmoVisionBackbone(config.build_vision_config(), llm_config, device)
        self.proprio_projector = (
            ProprioProjector(config.d_model, config.proprio_dim, device=device)
            # The released flow-matching checkpoints still project the state
            # into the main VLM prefix.  The action expert also consumes the
            # raw padded state, so these are two distinct conditioning paths.
            if config.use_proprio
            else None
        )
        if config.action_head == "l1_regression":
            self.action_head: nn.Module = L1RegressionActionHead(
                input_dim=config.d_model,
                hidden_dim=config.d_model,
                action_dim=config.fixed_action_dim,
                action_token_dim=config.action_token_dim,
                num_actions_chunk=config.num_actions_chunk,
                device=device,
            )
        elif config.action_head == "diffusion":
            self.action_head = DiffusionTransformerActionHead(config, device=device)
        elif config.action_head == "flow_matching":
            self.action_head = FlowMatchingActionHead(config, device=device)
        else:  # guarded by A1Config.validate
            raise ValueError(config.action_head)

    @staticmethod
    def _attention_bias(
        attention_mask: torch.Tensor,
        dtype: torch.dtype,
        *,
        causal: bool,
    ) -> torch.Tensor:
        batch, length = attention_mask.shape
        valid = attention_mask.to(torch.bool)
        allowed = valid[:, None, None, :].expand(batch, 1, length, length)
        if causal:
            triangle = torch.ones(length, length, dtype=torch.bool, device=valid.device).tril()
            allowed = allowed & triangle.reshape(1, 1, length, length)
        minimum = torch.finfo(dtype).min
        return torch.where(
            allowed,
            torch.zeros((), dtype=dtype, device=valid.device),
            torch.full((), minimum, dtype=dtype, device=valid.device),
        )

    def _embed_images(
        self,
        embeddings: torch.Tensor,
        images: torch.Tensor,
        image_masks: torch.Tensor | None,
        image_pooling: torch.Tensor,
        image_input_idx: torch.Tensor,
    ) -> torch.Tensor:
        features = self.vision_backbone(images, image_masks, image_pooling)
        batch = embeddings.shape[0]
        features = features.reshape(batch, -1, features.shape[-1]).to(device=embeddings.device, dtype=embeddings.dtype)
        indices = image_input_idx.reshape(batch, -1).to(embeddings.device)
        if features.shape[:2] != indices.shape:
            raise ValueError(f"A1 image feature/index mismatch: {tuple(features.shape[:2])} vs {tuple(indices.shape)}")
        valid = indices >= 0
        rows = torch.arange(batch, device=embeddings.device).unsqueeze(1).expand_as(indices)
        embeddings = embeddings.clone()
        embeddings[rows[valid], indices[valid]] += features[valid]
        return embeddings

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
        image_masks: torch.Tensor | None,
        image_pooling: torch.Tensor,
        image_input_idx: torch.Tensor,
        action_proprio: torch.Tensor | None,
        proprio_token_idx: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> A1ModelOutput:
        safe_ids = torch.where(input_ids >= 0, input_ids, 0)
        x = self.transformer.wte(safe_ids)
        if self.proprio_projector is not None:
            if action_proprio is None or proprio_token_idx is None:
                raise ValueError("A1 checkpoint requires state and its proprio token index")
            projected = self.proprio_projector(action_proprio.to(self.proprio_projector.fc1.weight.dtype)).to(x.dtype)
            rows = torch.arange(x.shape[0], device=x.device)
            x = x.clone()
            x[rows, proprio_token_idx.to(x.device)] = projected
        x = self._embed_images(x, images, image_masks, image_pooling, image_input_idx)
        x = self.transformer.emb_drop(x)
        if position_ids is None:
            position_ids = torch.clamp(attention_mask.long().cumsum(-1) - 1, min=0)
        causal = self.config.llm_causal_attention or self.config.action_head == "flow_matching"
        bias = self._attention_bias(attention_mask, x.dtype, causal=causal)
        caches: list[tuple[torch.Tensor, torch.Tensor]] | None = [] if use_cache else None
        for block in self.transformer.blocks:
            block_device = next(block.parameters()).device
            if x.device != block_device:
                x = x.to(block_device)
            block_bias = bias.to(block_device)
            block_positions = position_ids.to(block_device)
            x, cache = block(
                x,
                attention_bias=block_bias,
                position_ids=block_positions,
                layer_past=None,
                use_cache=use_cache,
            )
            if caches is not None:
                if cache is None:
                    raise RuntimeError("A1 Molmo block did not produce the requested KV cache")
                caches.append(cache)
        final_device = self.transformer.ln_f.weight.device
        x = self.transformer.ln_f(x.to(final_device))
        return A1ModelOutput(x, caches)

    def extract_action_hidden_states(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        action_start_token_id: int,
        action_end_token_id: int,
    ) -> torch.Tensor:
        expected = self.config.num_actions_chunk * self.config.action_token_dim
        selected = []
        for batch_index in range(input_ids.shape[0]):
            starts = torch.nonzero(input_ids[batch_index] == action_start_token_id, as_tuple=False).flatten()
            ends = torch.nonzero(input_ids[batch_index] == action_end_token_id, as_tuple=False).flatten()
            if not len(starts) or not len(ends):
                raise ValueError("A1 input is missing action boundary tokens")
            start = int(starts[-1])
            valid_ends = ends[ends > start]
            if not len(valid_ends):
                raise ValueError("A1 action end token occurs before its action start token")
            end = int(valid_ends[0])
            if end - start - 1 != expected:
                raise ValueError(f"A1 expected {expected} action tokens between boundaries, got {end - start - 1}")
            selected.append(hidden_states[batch_index, start + 1 : end])
        return torch.stack(selected, dim=0)

    @torch.no_grad()
    def predict_actions(
        self,
        batch: dict[str, torch.Tensor],
        *,
        action_start_token_id: int,
        action_end_token_id: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if self.config.action_head == "flow_matching":
            output = self.forward(**batch, use_cache=True)
            if output.attn_key_values is None or batch.get("action_proprio") is None:
                raise RuntimeError("A1 flow matching requires prefix KV and state")
            action_head = self.action_head
            assert isinstance(action_head, FlowMatchingActionHead)
            dtype = output.last_hidden_state.dtype
            action = torch.randn(
                batch["input_ids"].shape[0],
                self.config.num_actions_chunk,
                self.config.fixed_action_dim,
                device=output.last_hidden_state.device,
                dtype=dtype,
                generator=generator,
            )
            valid_lengths = batch["attention_mask"].sum(-1)
            state = batch["action_proprio"].to(action.device)
            steps = self.config.num_diffusion_inference_steps
            for index in range(steps):
                timestep = torch.full(
                    (action.shape[0],),
                    1.0 - index / steps,
                    device=action.device,
                    dtype=dtype,
                )
                velocity = action_head.predict_vector_field(
                    output.attn_key_values,
                    state,
                    action,
                    timestep,
                    valid_prefix_lengths=valid_lengths,
                )
                action = action - velocity / steps
            return action

        output = self.forward(**batch, use_cache=False)
        hidden = self.extract_action_hidden_states(
            output.last_hidden_state,
            batch["input_ids"].to(output.last_hidden_state.device),
            action_start_token_id=action_start_token_id,
            action_end_token_id=action_end_token_id,
        )
        if isinstance(self.action_head, L1RegressionActionHead):
            return self.action_head.predict_action(hidden)
        if isinstance(self.action_head, DiffusionTransformerActionHead):
            return self.action_head.condition_sampling(hidden, generator=generator)
        raise TypeError(f"Unexpected A1 action head: {type(self.action_head).__name__}")

    def apply_compile(self, *, mode: str = "max-autotune") -> None:
        """Compile stable repeated kernels while leaving preprocessing dynamic."""

        for block in self.transformer.blocks:
            block.compile(mode=mode, fullgraph=False)
        for block in self.vision_backbone.image_vit.transformer.resblocks:
            block.compile(mode=mode, fullgraph=False)
        if isinstance(self.action_head, DiffusionTransformerActionHead):
            for block in self.action_head.model.blocks:
                block.compile(mode=mode, fullgraph=False)


__all__ = [
    "A1Config",
    "A1ModelOutput",
    "AffordVLA",
    "DiT",
    "DiTBlock",
    "DiffusionTransformerActionHead",
    "FlowMatchingActionHead",
    "L1RegressionActionHead",
    "ProprioProjector",
]
