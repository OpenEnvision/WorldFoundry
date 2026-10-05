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

import torch

from ...components import ComponentBuildContext
from ...contracts import MultiModalDenoiserInput, MultiModalDenoiserOutput
from ...loaders import ModuleLoadSpec, NativeModuleLoader, checkpoint_json_config
from ..networks.cosmos3.model import Cosmos3OmniTransformer
from ..representations.cosmos3 import build_cosmos3_sequence_layout


class Cosmos3JointDenoiser:
    """Map framework modality states onto the Cosmos3 omni transformer.

    The runner owns ``ModalityState.latent`` / masks; this class only
    flattens them through the published packing rules and re-keys the
    transformer outputs as ``MultiModalDenoiserOutput.samples``.
    """

    def __init__(self, model: Cosmos3OmniTransformer) -> None:
        self.model = model

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
        layout = build_cosmos3_sequence_layout(
            input_ids,
            model_input.modalities,
            self.model.config,
            fps=fps,
        ).values
        device = video.latent.device
        sound = model_input.modalities.get("sound")
        action = model_input.modalities.get("action")
        domain_id = model_input.conditioning.get("action_domain_id")
        if action is not None and not isinstance(domain_id, torch.Tensor):
            raise TypeError("Cosmos3 action generation requires an action_domain_id tensor")

        vision, sound_prediction, action_prediction = self.model(
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
    return Cosmos3JointDenoiser(model)


__all__ = ["Cosmos3JointDenoiser", "build_cosmos3_joint_denoiser"]
