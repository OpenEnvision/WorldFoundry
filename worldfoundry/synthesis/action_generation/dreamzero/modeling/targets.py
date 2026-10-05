"""Rewrite serialized upstream Hydra targets to in-tree inference modules."""

from __future__ import annotations

from collections.abc import Mapping


_TARGET_PREFIXES = (
    (
        "groot.vla.model.n1_5.modules.cross_attention_dit",
        "worldfoundry.synthesis.action_generation.gr00t.modeling.dit",
    ),
    (
        "groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk",
        "worldfoundry.base_models.diffusion_model.models.networks.wan.variants.dreamzero.action",
    ),
    (
        "groot.vla.model.dreamzero.modules.wan_video_text_encoder",
        "worldfoundry.base_models.diffusion_model.models.encoders.wan.model",
    ),
    (
        "groot.vla.model.dreamzero.modules.wan_video_image_encoder",
        "worldfoundry.base_models.diffusion_model.models.encoders.wan.image",
    ),
    (
        "groot.vla.model.dreamzero.modules.wan_video_vae",
        "worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model",
    ),
    (
        "groot.vla.model.dreamzero.modules.wan_video_dit",
        "worldfoundry.base_models.diffusion_model.models.networks.wan.model",
    ),
    (
        "groot.vla.model.dreamzero.modules.wan_video_camera_controller",
        "worldfoundry.base_models.diffusion_model.models.networks.wan.adapter",
    ),
    (
        "groot.vla.model.dreamzero.modules.flow_match_scheduler",
        "worldfoundry.core.nn.diffusion.schedulers",
    ),
    (
        "groot.vla.model.dreamzero.modules.flow_unipc_multistep_scheduler",
        "worldfoundry.base_models.diffusion_model.schedulers.flow_unipc",
    ),
    (
        "groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf",
        "worldfoundry.synthesis.action_generation.dreamzero.modeling.action_head",
    ),
    (
        "groot.vla.model.dreamzero.backbone.identity",
        "worldfoundry.synthesis.action_generation.dreamzero.modeling.backbone",
    ),
    (
        "groot.vla.model.dreamzero.base_vla",
        "worldfoundry.synthesis.action_generation.dreamzero.modeling.vla",
    ),
    (
        "groot.vla.model.dreamzero.transform.dreamzero_cotrain",
        "worldfoundry.synthesis.action_generation.dreamzero.preprocessing",
    ),
    (
        "groot.vla.data.transform",
        "worldfoundry.synthesis.action_generation.dreamzero.preprocessing",
    ),
    (
        "groot.vla.data.dataset",
        "worldfoundry.synthesis.action_generation.dreamzero.preprocessing",
    ),
)


_ALLOWED_INFERENCE_TARGET_PREFIXES = (
    "worldfoundry.synthesis.action_generation.dreamzero.",
    "worldfoundry.synthesis.action_generation.gr00t.modeling.dit.",
    "worldfoundry.base_models.diffusion_model.models.networks.wan.",
    "worldfoundry.base_models.diffusion_model.models.encoders.wan.",
    "worldfoundry.base_models.diffusion_model.models.autoencoders.wan.",
    "worldfoundry.base_models.diffusion_model.schedulers.flow_unipc.",
    "worldfoundry.core.nn.diffusion.schedulers.",
)


def rewrite_target_path(target: str) -> str:
    for upstream, in_tree in _TARGET_PREFIXES:
        if target.startswith(upstream):
            return in_tree + target[len(upstream) :]
    return target


def rewrite_targets(value) -> None:
    """Mutate dict/OmegaConf containers without resolving missing training fields."""
    if isinstance(value, Mapping) or hasattr(value, "keys"):
        for key in list(value.keys()):
            try:
                child = value[key]
            except Exception:
                continue
            if key == "_target_" and isinstance(child, str):
                value[key] = rewrite_target_path(child)
            else:
                rewrite_targets(child)
    elif isinstance(value, list) or value.__class__.__name__ == "ListConfig":
        for child in value:
            rewrite_targets(child)


def validate_inference_targets(value, *, path: str = "config") -> None:
    """Reject checkpoint-controlled Hydra targets outside the inference tree.

    DreamZero checkpoints include Hydra configuration.  Rewriting known upstream
    paths is not sufficient on its own: an untrusted checkpoint could otherwise
    name an arbitrary callable.  Every configuration fragment passed to
    ``hydra.utils.instantiate`` is checked immediately before construction.
    """

    if isinstance(value, Mapping) or hasattr(value, "keys"):
        try:
            keys = list(value.keys())
        except Exception as error:
            raise TypeError(f"DreamZero configuration is not inspectable at {path}") from error
        for key in keys:
            try:
                child = value[key]
            except Exception as error:
                raise TypeError(
                    f"DreamZero configuration value is not inspectable at {path}.{key}"
                ) from error
            child_path = f"{path}.{key}"
            if key == "_target_":
                if not isinstance(child, str):
                    raise TypeError(f"DreamZero Hydra target must be a string at {child_path}")
                if "${" in child or not child.startswith(_ALLOWED_INFERENCE_TARGET_PREFIXES):
                    raise ValueError(
                        f"DreamZero Hydra target is not allowlisted at {child_path}: {child}"
                    )
            else:
                validate_inference_targets(child, path=child_path)
    elif isinstance(value, list) or value.__class__.__name__ == "ListConfig":
        for index, child in enumerate(value):
            validate_inference_targets(child, path=f"{path}[{index}]")
