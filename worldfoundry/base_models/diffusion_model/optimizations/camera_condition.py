# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""Request-owned, timestep-independent camera conditioning.

Prepared tensors are explicit forward inputs, never model-global state.
Projection and reuse follow the denoiser's placement and autocast policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from einops import rearrange
from torch.nn import functional as F


@dataclass
class CameraConditionCache:
    """One request's fixed camera input and lazily projected features.

    Projection runs inside the model forward, after its placement/offload hooks
    and under the same autocast policy as the uncached path.
    """

    source: torch.Tensor
    features: torch.Tensor | None = None
    _batches: dict = field(default_factory=dict, repr=False)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, selection: slice):
        """Keep each CFG batch slice's original projection shape and arithmetic."""
        key = (selection.start, selection.stop, selection.step)
        if key not in self._batches:
            self._batches[key] = CameraConditionCache(self.source[selection])
        return self._batches[key]

    @property
    def nbytes(self):
        own = 0 if self.features is None else self.features.numel() * self.features.element_size()
        return own + sum(batch.nbytes for batch in self._batches.values())


def camera_features(module, condition, *, encode: Callable | None = None):
    """Reuse a request-owned camera projection; raw tensors retain autograd."""
    project = module if encode is None else encode
    if not isinstance(condition, CameraConditionCache):
        return project(condition)
    if module.training or torch.is_grad_enabled():
        raise RuntimeError("camera feature reuse requires eval no-grad inference")
    if condition.features is None:
        condition.features = project(condition.source)
    return condition.features


def _pad_tokens(camera, length):
    if length is not None and camera.shape[1] < length:
        camera = torch.cat((camera, camera.new_zeros(camera.shape[0], length - camera.shape[1], camera.shape[2])), dim=1)
    return camera


def embed_wan_camera_condition(model, cameras, *, token_length=None, pad_before_embedding=False):
    """Canonical camera patch projection, including Action-22's raw-token padding."""
    tokens = [rearrange(value, "1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)",
                        c1=model.patch_size[0], c2=model.patch_size[1], c3=model.patch_size[2]) for value in cameras]
    camera = torch.cat(tokens, dim=1)
    if pad_before_embedding:
        camera = _pad_tokens(camera, token_length)
    camera = model.patch_embedding_wancamctrl(camera)
    camera = camera + model.c2ws_hidden_states_layer2(F.silu(model.c2ws_hidden_states_layer1(camera)))
    return camera if pad_before_embedding else _pad_tokens(camera, token_length)


def prepare_wan_camera_condition(model, condition, *, token_length=None, pad_before_embedding=False):
    """Project camera tokens and per-block affine terms for one inference chunk."""
    if model.training or torch.is_grad_enabled():
        raise RuntimeError("camera preparation requires eval no-grad inference")
    camera = embed_wan_camera_condition(model, condition["c2ws_plucker_emb"],
                                       token_length=token_length, pad_before_embedding=pad_before_embedding)
    modulations = tuple(project_camera_modulation(block, camera) for block in model.blocks)
    return {**condition, "c2ws_plucker_emb": camera, "_camera_modulations": modulations}


def prepare_action22_camera_condition(model, plucker_emb, *, token_length):
    """Normalize Action-22 / Matrix-Game camera inputs into the shared chunk cache."""
    if torch.is_tensor(plucker_emb):
        cameras = [value.unsqueeze(0) for value in plucker_emb]
    else:
        cameras = [value.unsqueeze(0) if value.ndim == 4 else value for value in plucker_emb]
    return prepare_wan_camera_condition(model, {"c2ws_plucker_emb": cameras},
                                       token_length=token_length, pad_before_embedding=True)


def project_camera_modulation(block, camera):
    """Preserve the checkpoint's camera MLP and residual arithmetic."""
    hidden = block.cam_injector_layer2(F.silu(block.cam_injector_layer1(camera))) + camera
    return block.cam_scale_layer(hidden), block.cam_shift_layer(hidden)


def camera_condition_for_block(condition, index):
    """Bind one block's prepared terms without mutating caller-owned inputs."""
    if condition is None or "_camera_modulations" not in condition:
        return condition
    return {**condition, "_camera_modulation": condition["_camera_modulations"][index]}


def apply_camera_modulation(block, hidden, condition):
    """Apply prepared affine terms or compute the original inference/training path."""
    if condition is None or "c2ws_plucker_emb" not in condition:
        return hidden
    if "_camera_modulation" in condition:
        if block.training or torch.is_grad_enabled():
            raise RuntimeError("prepared camera modulation is inference-only")
        scale, shift = condition["_camera_modulation"]
    else:
        scale, shift = project_camera_modulation(block, condition["c2ws_plucker_emb"])
    return (1.0 + scale) * hidden + shift
