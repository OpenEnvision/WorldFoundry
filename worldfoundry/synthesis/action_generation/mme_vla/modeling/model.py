# ruff: noqa: F722 - jaxtyping dimension strings in annotations.
import dataclasses
from typing import Any

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from ...openpi import array_typing as at
from ...openpi.modeling import siglip as _siglip
from ...openpi.modeling.model import Actions, BaseModel
from ...openpi.modeling.pi0 import posemb_sincos
from ...openpi.modeling.pi0_config import Pi0Config
from ..history_config import get_history_config
from . import history_gemma as _gemma
from .observation import HistAugObservation, preprocess_observation


def make_attn_mask(input_mask, mask_ar, mask_na=None):
    """Adapted from pi0.py

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    mask_na: bool[B, N]
       [[0 0 0 1 1 1 0 0 0 ...]]: The 1s tokens can not attend to the first three 0s.
       this is used for vision tokens not attend to memory tokens, but action tokens can

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    if mask_na is not None:
        mask_na = jnp.broadcast_to(mask_na, input_mask.shape)
        mask_not_attend = jnp.logical_and(
            jnp.logical_or(mask_na[:, None, :], mask_na[:, :, None]),
            einops.repeat(jnp.cumsum(mask_na, axis=1) <= 0, "b k -> b s k", s=mask_na.shape[1]),
        )
        # print(mask_not_attend)
        return jnp.where(mask_not_attend, False, jnp.logical_and(attn_mask, valid_mask))
    else:
        return jnp.logical_and(attn_mask, valid_mask)


@dataclasses.dataclass(frozen=True)
class HistoryPi0Config(Pi0Config):
    # paligemma_variant: _gemma.Variant = "gemma_2b_lora"
    # action_expert_variant: _gemma.Variant = "gemma_300m"
    memory_expert_variant: _gemma.Variant = "gemma_150m"

    use_history: bool = False  # Use history or not
    history_config: str | None = None  # history config
    max_token_len: int = 64

    @override
    def create(self, rng: at.KeyArrayLike) -> "HistoryPi0":
        # Load the history config if it's specified
        if self.history_config is not None:
            loaded_config = get_history_config(self.history_config)
            # Create a new config with the loaded history config
            config_with_loaded_history = dataclasses.replace(self, history_config=loaded_config)

            max_token_len = self.max_token_len
            if loaded_config.representation_type == "symbolic":
                if loaded_config.symbolic_memory.type in ["simple_subgoal", "grounded_subgoal"]:
                    max_token_len *= 2
                else:
                    raise ValueError(f"Not supported symbolic memory type: {loaded_config.symbolic_memory.type}")
                config_with_loaded_history = dataclasses.replace(
                    config_with_loaded_history, max_token_len=max_token_len
                )
                print("symbolic_memory_type: ", loaded_config.symbolic_memory.type)

            print("max_token_len: ", config_with_loaded_history.max_token_len)

            return HistoryPi0(config_with_loaded_history, rngs=nnx.Rngs(rng))
        else:
            return HistoryPi0(self, rngs=nnx.Rngs(rng))

    def inputs_spec(self, *, batch_size: int = 1) -> tuple[HistAugObservation, Actions]:
        base_obs_spec, action_spec = super().inputs_spec(batch_size=batch_size)
        with at.disable_typechecking():
            if not self.use_history:
                observation_spec = base_obs_spec  # basic pi0
            else:
                if self.history_config.representation_type == "symbolic":
                    observation_spec = HistAugObservation.from_base_obs(
                        base_obs_spec,
                        symbolic_tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                        symbolic_tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                    )
                elif self.history_config.representation_type == "perceptual":
                    observation_spec = HistAugObservation.from_base_obs(
                        base_obs_spec,
                        static_image_emb=jax.ShapeDtypeStruct(
                            [
                                batch_size,
                                self.history_config.budget,
                                self.history_config.memory_feature.img.input_dim,
                            ],
                            jnp.float32,
                        ),
                        static_mask=jax.ShapeDtypeStruct(
                            [batch_size, self.history_config.budget],
                            jnp.bool_,
                        ),
                        static_pos_emb=jax.ShapeDtypeStruct(
                            [
                                batch_size,
                                self.history_config.budget,
                                self.history_config.memory_feature.pos.input_dim,
                            ],
                            jnp.float32,
                        ),
                        static_state_emb=jax.ShapeDtypeStruct(
                            [
                                batch_size,
                                self.history_config.budget,
                                self.history_config.memory_feature.state.input_dim,
                            ],
                            jnp.float32,
                        ),
                    )
                elif self.history_config.representation_type == "recurrent":
                    observation_spec = HistAugObservation.from_base_obs(
                        base_obs_spec,
                        recur_image_emb=jax.ShapeDtypeStruct(
                            [
                                batch_size,
                                self.history_config.recurrent_memory.max_recur_steps,
                                self.history_config.num_views,
                                self.history_config.token_per_image,
                                self.history_config.memory_feature.img.input_dim,
                            ],
                            jnp.float32,
                        ),
                        recur_mask=jax.ShapeDtypeStruct(
                            [
                                batch_size,
                                self.history_config.recurrent_memory.max_recur_steps,
                            ],
                            jnp.bool_,
                        ),
                        recur_pos_emb=jax.ShapeDtypeStruct(
                            [
                                batch_size,
                                self.history_config.recurrent_memory.max_recur_steps,
                                self.history_config.num_views,
                                self.history_config.token_per_image,
                                self.history_config.memory_feature.pos.input_dim,
                            ],
                            jnp.float32,
                        ),
                        recur_state_emb=jax.ShapeDtypeStruct(
                            [
                                batch_size,
                                self.history_config.recurrent_memory.max_recur_steps,
                                self.history_config.memory_feature.state.input_dim,
                            ],
                            jnp.float32,
                        ),
                    )

                else:
                    raise ValueError(f"Not supported representation type: {self.history_config.representation_type}")

        return observation_spec, action_spec


class HistoryPi0(BaseModel):
    def __init__(self, config: HistoryPi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.config = config
        self.use_history = config.use_history

        if self.use_history:
            self.history_config = config.history_config
            self.integration_type = config.history_config.integration_type
            self.representation_type = config.history_config.representation_type
            assert self.integration_type in ["context", "modulation", "expert"]
            assert self.representation_type in ["perceptual", "recurrent", "symbolic"]

            if self.representation_type == "perceptual":
                from .perceptual_memory import PerceptualMemory

                self.mem_encoder = PerceptualMemory(
                    config=self.history_config,
                    rngs=rngs,
                    dtype=config.dtype,
                )
            elif self.representation_type == "recurrent":
                from .recurrent_memory import RecurrentMemory

                self.mem_encoder = RecurrentMemory(
                    config=self.history_config,
                    rngs=rngs,
                    dtype=config.dtype,
                )
            elif self.representation_type == "symbolic":
                self.integration_type = None  # if symbolic, we only use it as languge input
            else:
                raise ValueError(f"Not supported representation type: {self.representation_type}")

            print(
                f"====== Using History, Representation Type: {self.representation_type} , Integration Type: {self.integration_type} ======"
            )
            if self.representation_type == "perceptual":
                print(f"Perceptual Memory using {self.history_config.perceptual_memory.type} type\n")
            elif self.representation_type == "recurrent":
                print(f"Recurrent Memory using {self.history_config.recurrent_memory.type} type\n")
            else:
                print("\n")

            if self.integration_type == "expert":
                memory_expert_config = _gemma.get_config(config.memory_expert_variant)
                llm = nnx_bridge.ToNNX(
                    _gemma.Module(
                        configs=[
                            memory_expert_config,
                            paligemma_config,
                            action_expert_config,
                        ],
                        embed_dtype=config.dtype,
                        adarms=config.pi05,
                        integration_type=self.integration_type,
                    )
                )
                llm.lazy_init(
                    rngs=rngs,
                    method="init",
                    use_adarms=([False, False, True] if config.pi05 else [False, False, False]),
                    mem_mods=[False, False, False],
                )
            else:
                llm = nnx_bridge.ToNNX(
                    _gemma.Module(
                        configs=[paligemma_config, action_expert_config],
                        embed_dtype=config.dtype,
                        adarms=config.pi05,
                        integration_type=self.integration_type,
                    )
                )
                llm.lazy_init(
                    rngs=rngs,
                    method="init",
                    use_adarms=[False, True] if config.pi05 else [False, False],
                    mem_mods=[False, True] if self.integration_type == "modulation" else [False, False],
                )

        else:
            # safe setting
            self.history_config = self.integration_type = self.representation_type = None

            llm = nnx_bridge.ToNNX(
                _gemma.Module(
                    configs=[paligemma_config, action_expert_config],
                    embed_dtype=config.dtype,
                    adarms=config.pi05,
                    integration_type=self.integration_type,
                )
            )
            llm.lazy_init(
                rngs=rngs,
                method="init",
                use_adarms=[False, True] if config.pi05 else [False, False],
                mem_mods=[False, False],
            )

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)

        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

    @at.typecheck
    def embed_memory(self, obs: HistAugObservation):
        if self.representation_type == "perceptual":
            tokens, _, stats = self.mem_encoder(obs.static_image_emb, obs.static_pos_emb, obs.static_state_emb)
            input_mask = obs.static_mask
            ar_mask = [False] * tokens.shape[1]
            na_mask = [False] * tokens.shape[1]
        elif self.representation_type == "recurrent":
            (tokens, input_mask), _, stats = self.mem_encoder(
                obs.recur_image_emb, obs.recur_mask, obs.recur_pos_emb, obs.recur_state_emb
            )
            ar_mask = [False] * tokens.shape[1]
            na_mask = [False] * tokens.shape[1]
        else:
            tokens = None
            input_mask = None
            ar_mask = None
            na_mask = None
            stats = None
        return tokens, input_mask, ar_mask, na_mask, stats

    @at.typecheck
    def embed_prefix(
        self, obs: HistAugObservation
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Bool[at.Array, " s"],
        Any | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        na_mask = []

        if self.integration_type == "context":
            (
                mem_tokens,
                mem_input_mask,
                mem_ar_mask,
                mem_na_mask,
                stats,
            ) = self.embed_memory(obs)
            if mem_tokens is not None:
                tokens.append(mem_tokens)
                input_mask.append(mem_input_mask)
                ar_mask += mem_ar_mask
                na_mask += mem_na_mask
        else:
            stats = None

        # embed images
        for i, name in enumerate(obs.images):
            image_tokens, _ = self.PaliGemma.img(obs.images[name])

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            if i == 0:
                ar_mask += [True] + ([False] * (image_tokens.shape[1] - 1))
            else:
                ar_mask += [False] * image_tokens.shape[1]
            na_mask += [True] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if self.use_history and self.representation_type == "symbolic":
            tokenized_inputs = self.PaliGemma.llm(obs.symbolic_tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.symbolic_tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
            na_mask += [False] * tokenized_inputs.shape[1]

        elif obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
            na_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        na_mask = jnp.array(na_mask)

        return tokens, input_mask, ar_mask, na_mask, stats

    @at.typecheck
    def embed_suffix(
        self,
        obs: HistAugObservation,
        noisy_actions: Actions,
        timestep: at.Float[at.Array, " b"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        na_mask = []

        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]
            na_mask += [False]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        na_mask += [False] * self.action_horizon
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)

        ar_mask = jnp.array(ar_mask)
        na_mask = jnp.array(na_mask)
        return tokens, input_mask, ar_mask, na_mask, adarms_cond

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: HistAugObservation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> Actions:
        observation = preprocess_observation(observation)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        if self.integration_type == "expert":
            mem_tokens, mem_input_mask, mem_ar_mask, mem_na_mask, _ = self.embed_memory(observation)
            vlm_tokens, vlm_mask, vlm_ar_mask, vlm_na_mask, _ = self.embed_prefix(observation)
            mem_ar_mask = jnp.array(mem_ar_mask)
            mem_na_mask = jnp.array(mem_na_mask)
            prefix_mask = jnp.concatenate([mem_input_mask, vlm_mask], axis=1)
            prefix_ar_mask = jnp.concatenate([mem_ar_mask, vlm_ar_mask], axis=0)
            prefix_na_mask = jnp.concatenate([mem_na_mask, vlm_na_mask], axis=0)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask, prefix_na_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            _, kv_cache = self.PaliGemma.llm(
                [mem_tokens, vlm_tokens, None],
                mask=prefix_attn_mask,
                positions=positions,
            )

        elif self.integration_type == "modulation":
            prefix_tokens, prefix_mask, prefix_ar_mask, _, _ = self.embed_prefix(observation)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
            mem_seq, mem_mask, _, _, _ = self.embed_memory(observation)

        else:
            prefix_tokens, prefix_mask, prefix_ar_mask, prefix_na_mask, _ = self.embed_prefix(observation)
            if self.integration_type == "context":
                prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask, prefix_na_mask)
            else:
                prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, _, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_mask.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            if self.integration_type == "expert":
                (mem_out, prefix_out, suffix_out), _ = self.PaliGemma.llm(
                    [None, None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, None, adarms_cond],
                )
                assert mem_out is None
            elif self.integration_type == "modulation":
                (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, adarms_cond],
                    mem_seq=[None, mem_seq],
                    mem_mask=[None, mem_mask],
                )
            else:
                (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, adarms_cond],
                )

            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def vision_encode(self, images: at.Float[at.Array, "k v 224 224 3"]):
        shape = images.shape
        assert shape[-3:] == (
            224,
            224,
            3,
        ), f"x.shape: {shape}, should be (B, 224, 224, 3)"

        x = einops.rearrange(images, "k v h w c -> (k v) h w c")
        out, _ = self.PaliGemma.img(x)
        out = einops.rearrange(out, "(k v) p d -> k v p d", k=shape[0], v=shape[1])

        return out
