"""Cosmos3 joint (video / sound / action) denoiser adapter.

Implements the multimodal denoiser surface used by the joint-multistage
runner: ``__call__(MultiModalDenoiserInput) -> MultiModalDenoiserOutput``.
This is not the single-tensor :class:`~...contracts.Denoiser` Protocol;
Cosmos3 packs several :class:`~...contracts.ModalityState` objects into one
omni sequence and returns a sample per requested modality.

``MultiModalDenoiserInput.conditioning`` must contain tokenizer ``input_ids``
and, when action tokens are present, ``action_domain_id``.  Optional
``fps`` defaults to 24.  Video is required; sound and action are optional
and only forwarded when their modality state is present.

Layout construction is delegated to
:func:`~..representations.cosmos3.build_cosmos3_sequence_layout` so this
adapter stays a thin mapping from framework-owned states onto
:class:`~..networks.cosmos3.model.Cosmos3OmniTransformer` kwargs.  The
inner transformer is **not** CUDA-graph safe (variable-length packing).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ...components import ComponentBuildContext
from ...contracts import MultiModalDenoiserInput, MultiModalDenoiserOutput
from ...loaders import ModuleLoadSpec, NativeModuleLoader, checkpoint_json_config
from ..networks.cosmos3.acceleration import (
    Cosmos3AccelerationConfig,
    accelerate_cosmos3,
    cosmos3_runtime_receipt,
)
from ..networks.cosmos3.model import Cosmos3OmniTransformer
from ..representations.cosmos3 import build_cosmos3_sequence_layout
from .graph_wrapped import (
    FeatureCacheDenoiserMixin,
    GraphWrappedDenoiserMixin,
    resolve_cuda_graph_mode,
)


class Cosmos3JointDenoiser(GraphWrappedDenoiserMixin, FeatureCacheDenoiserMixin):
    """Map framework modality states onto the Cosmos3 omni transformer.

    The runner owns ``ModalityState.latent`` / masks; this class flattens them
    through the published packing rules, caches static per-request layouts, and
    forwards SOL-Engine step metadata (exact ``und_seq`` static KV cache,
    TeaCache/FirstBlock, step-selective FP8/NVFP4, and Piecewise CUDA Graph).
    """

    def __init__(
        self,
        model: Cosmos3OmniTransformer,
        *,
        cuda_graph: bool | str = False,
        extra_key: str = "cosmos3",
    ) -> None:
        self.model = model
        mode = (
            "piecewise"
            if isinstance(cuda_graph, str) and cuda_graph.strip().casefold() == "piecewise"
            else (
                "whole"
                if bool(cuda_graph) and str(cuda_graph).strip().casefold() not in {"off", "none", "false"}
                else "off"
            )
        )
        if mode == "whole":
            raise ValueError(
                "Cosmos3JointDenoiser does not support whole-model CUDA Graphs (variable-length packing); "
                "use cuda_graph='piecewise' or 'off'"
            )
        existing_piecewise = getattr(model, "_cosmos3_piecewise_graph", None)
        if mode == "off" and existing_piecewise is not None:
            mode = "piecewise"
        if mode == "piecewise" and existing_piecewise is not None:
            self._init_graph_runner(model, enabled=False, extra_key=extra_key)
            self._piecewise_graph = existing_piecewise
            self._piecewise_graph_extra_key = extra_key
        else:
            self._init_graph_runner(
                model,
                enabled=mode if mode != "off" else False,
                extra_key=extra_key,
            )
        if self._piecewise_graph is not None:
            model._cosmos3_piecewise_graph = self._piecewise_graph
        self._layout_cache: dict[
            tuple[Any, ...],
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor | None,
                torch.Tensor | None,
                Mapping[str, Any],
            ],
        ] = {}
        self._last_request_id: str | None = None
        self._seen_step0_branches: set[str] = set()
        self._last_step_index: int | None = None

    def apply_sol_engine(self, config: Cosmos3AccelerationConfig | str = "sol_balanced") -> dict[str, Any]:
        """Install the Cosmos3 SOL-Engine acceleration stack on ``self.model``."""
        receipt = accelerate_cosmos3(self.model, config)
        self._graph_runner = None
        self._piecewise_graph = getattr(self.model, "_cosmos3_piecewise_graph", None)
        return receipt

    def end_request(self, request_id: str, *, error: BaseException | None = None) -> None:
        del error
        self.model.reset_sol_caches()
        self._layout_cache.clear()
        self._seen_step0_branches.clear()
        self._last_step_index = None
        if self._last_request_id == request_id:
            self._last_request_id = None

    def runtime_optimization_report(self, request_id: str | None = None) -> dict[str, Any]:
        del request_id
        sol_receipt = cosmos3_runtime_receipt(self.model)
        return {
            "cosmos3_sol_engine": sol_receipt,
            "cuda_graph": self.graph_report(),
            "fallbacks": list(sol_receipt.get("fallbacks", [])),
        }

    @staticmethod
    def _step_timesteps(
        count: int,
        timestep: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Broadcast the runner's scalar noise level to ``count`` packed tokens."""
        return torch.full(
            (count,),
            float(timestep.item()),
            device=device,
            dtype=torch.float32,
        )

    @staticmethod
    def _opt_tensor_equal(saved: torch.Tensor | None, current: torch.Tensor | None) -> bool:
        if saved is None or current is None:
            return saved is None and current is None
        return (
            saved.device == current.device
            and saved.dtype == current.dtype
            and saved.shape == current.shape
            and torch.equal(saved, current)
        )

    def __call__(self, model_input: MultiModalDenoiserInput) -> MultiModalDenoiserOutput:
        """Evaluate one joint denoise step for the requested modalities."""
        try:
            input_ids = model_input.conditioning["input_ids"]
            video = model_input.modalities["video"]
        except KeyError as error:
            raise KeyError(f"Cosmos3 denoising is missing {error.args[0]!r}") from error
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("Cosmos3 input_ids must be a tensor")
        fps = float(model_input.conditioning.get("fps", 24.0))
        request_id = getattr(model_input, "request_id", None) or model_input.conditioning.get("request_id")
        step_index = getattr(model_input, "step_index", None)
        total_steps = getattr(model_input, "total_steps", None)
        branch = str(getattr(model_input, "branch", "positive") or "positive")

        new_request_started = False
        if request_id is not None and request_id != self._last_request_id:
            self.model.reset_sol_caches()
            self._layout_cache.clear()
            self._seen_step0_branches.clear()
            self._last_request_id = str(request_id)
            new_request_started = True
        elif step_index == 0 and (
            branch in self._seen_step0_branches or (self._last_step_index is not None and self._last_step_index > 0)
        ):
            self.model.reset_sol_caches()
            self._seen_step0_branches.clear()
            new_request_started = True

        if step_index == 0:
            if new_request_started or not self._seen_step0_branches:
                self._begin_graph_request_window()
            self._seen_step0_branches.add(branch)
        if step_index is not None:
            self._last_step_index = int(step_index)

        sound = model_input.modalities.get("sound")
        action = model_input.modalities.get("action")
        sound_mask = sound.denoise_mask if sound is not None else None
        action_mask = action.denoise_mask if action is not None else None
        layout_key = (
            branch,
            tuple(input_ids.shape),
            tuple(video.latent.shape),
            tuple(video.denoise_mask.shape),
            tuple(sound.latent.shape) if sound is not None else None,
            tuple(sound_mask.shape) if sound_mask is not None else None,
            tuple(action.latent.shape) if action is not None else None,
            tuple(action_mask.shape) if action_mask is not None else None,
            fps,
        )
        cached_entry = self._layout_cache.get(layout_key)
        layout: Mapping[str, Any] | None = None
        if cached_entry is not None:
            saved_ids, saved_vmask, saved_smask, saved_amask, saved_layout = cached_entry
            if (
                self._opt_tensor_equal(saved_ids, input_ids)
                and self._opt_tensor_equal(saved_vmask, video.denoise_mask)
                and self._opt_tensor_equal(saved_smask, sound_mask)
                and self._opt_tensor_equal(saved_amask, action_mask)
            ):
                layout = saved_layout
        if layout is None:
            layout = build_cosmos3_sequence_layout(
                input_ids,
                model_input.modalities,
                self.model.config,
                fps=fps,
            ).values
            if len(self._layout_cache) >= 8:
                self._layout_cache.clear()
            self._layout_cache[layout_key] = (
                input_ids.detach().clone(),
                video.denoise_mask.detach().clone(),
                sound_mask.detach().clone() if sound_mask is not None else None,
                action_mask.detach().clone() if action_mask is not None else None,
                layout,
            )

        device = video.latent.device
        domain_id = model_input.conditioning.get("action_domain_id")
        if action is not None and not isinstance(domain_id, torch.Tensor):
            raise TypeError("Cosmos3 action generation requires an action_domain_id tensor")

        vision, sound_prediction, action_prediction = self._run_network(
            input_ids=layout["input_ids"],
            text_indexes=layout["text_indexes"],
            position_ids=layout["position_ids"],
            und_len=layout["und_len"],
            sequence_length=layout["sequence_length"],
            vision_tokens=[video.latent],
            vision_token_shapes=layout["vision_token_shapes"],
            vision_sequence_indexes=layout["vision_sequence_indexes"],
            vision_mse_loss_indexes=layout["vision_mse_loss_indexes"],
            vision_timesteps=self._step_timesteps(
                layout["num_noisy_vision_tokens"], model_input.timestep, device=device
            ),
            vision_noisy_frame_indexes=layout["vision_noisy_frame_indexes"],
            sound_tokens=[sound.latent] if sound is not None else None,
            sound_token_shapes=layout.get("sound_token_shapes"),
            sound_sequence_indexes=layout.get("sound_sequence_indexes"),
            sound_mse_loss_indexes=layout.get("sound_mse_loss_indexes"),
            sound_timesteps=(
                self._step_timesteps(layout["num_noisy_sound_tokens"], model_input.timestep, device=device)
                if sound is not None
                else None
            ),
            sound_noisy_frame_indexes=layout.get("sound_noisy_frame_indexes"),
            action_tokens=[action.latent] if action is not None else None,
            action_token_shapes=layout.get("action_token_shapes"),
            action_sequence_indexes=layout.get("action_sequence_indexes"),
            action_mse_loss_indexes=layout.get("action_mse_loss_indexes"),
            action_timesteps=(
                self._step_timesteps(layout["num_noisy_action_tokens"], model_input.timestep, device=device)
                if action is not None
                else None
            ),
            action_noisy_frame_indexes=layout.get("action_noisy_frame_indexes"),
            action_domain_ids=[domain_id.to(device)] if action is not None else None,
            step_index=step_index,
            total_steps=total_steps,
            cache_branch=branch,
        )
        samples = {"video": vision[0]}
        if sound is not None:
            if sound_prediction is None:
                raise RuntimeError("Cosmos3 transformer omitted the requested sound prediction")
            samples["sound"] = sound_prediction[0]
        if action is not None:
            if action_prediction is None:
                raise RuntimeError("Cosmos3 transformer omitted the requested action prediction")
            samples["action"] = action_prediction[0]
        return MultiModalDenoiserOutput(samples=samples)


def build_cosmos3_joint_denoiser(context: ComponentBuildContext) -> Cosmos3JointDenoiser:
    """Load Cosmos3 directly from its official sharded safetensors checkpoint."""

    from worldfoundry.core.vram import AutoWrappedLinear, AutoWrappedModule

    model = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=Cosmos3OmniTransformer,
            config_resolver=lambda checkpoint: checkpoint_json_config(checkpoint, "transformer/config.json"),
            vram_module_map={
                torch.nn.Embedding: AutoWrappedModule,
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.RMSNorm: AutoWrappedModule,
            },
            layer_container="layers",
        ),
        context.require_checkpoint("weights"),
        context.policy,
    )
    if not isinstance(model, Cosmos3OmniTransformer):
        raise TypeError(f"expected Cosmos3OmniTransformer, got {type(model).__name__}")
    graph_mode = resolve_cuda_graph_mode(context)
    sol_profile = context.component_options.get("sol_profile", context.policy.options.get("sol_profile"))
    denoiser = Cosmos3JointDenoiser(
        model,
        cuda_graph=graph_mode if graph_mode != "off" else False,
        extra_key="cosmos3",
    )
    if sol_profile is not None:
        denoiser.apply_sol_engine(sol_profile)
    return denoiser


__all__ = ["Cosmos3JointDenoiser", "build_cosmos3_joint_denoiser"]
