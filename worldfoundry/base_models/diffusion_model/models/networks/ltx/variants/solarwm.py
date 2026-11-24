# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: LicenseRef-LTX-2.x-Community
"""SolarWM keyframe activation on the shared LTX checkpoint graph.

Adapted from the Lightricks LTX-2 inference runtime; see THIRD-PARTY-NOTICES.
"""

import torch

from worldfoundry.base_models.diffusion_model.models.networks.ltx.model import LTXModel


class SolarWMModel(LTXModel):
    """Materialize optional keyframe embeddings when the loader enables them."""

    @property
    def supports_keyframes_abs_pos_embedding(self) -> bool:
        """Whether this model has a usable keyframe absolute-position embedding.
        False for models built without the flag, and also for a model whose config set the flag
        but whose checkpoint carried no weight for it (the parameter would still be on ``meta``).
        """
        embedding = self._keyframes_embedding()
        return embedding is not None and not embedding.is_meta

    def enable_keyframes_abs_pos_embedding(self) -> None:
        """Ensure the keyframe embedding exists and holds real zeros, creating it if needed.
        Covers both ways a loaded model can arrive without a usable parameter:
        - The checkpoint predates the feature entirely, so its config never set
          ``use_keyframes_abs_pos_embedding`` and no parameter was built. It is created here.
        - The config did set the flag but the checkpoint carried no weight for it. Models are
          built on the meta device and loaded with ``strict=False, assign=True``, so such a
          parameter stays on ``meta`` and would fail at the first forward.
        Idempotent, and never overwrites a parameter that already holds real storage -- a
        checkpoint carrying a trained embedding keeps it. The preprocessors resolve the parameter
        through a provider on every call, so enabling it after the model is built is safe.
        A zero embedding is an exact no-op, so this only makes the marker *harmless*, not
        meaningful: it does not give an untrained checkpoint the keyframe capability.
        """
        if not self.model_type.is_video_enabled():
            raise ValueError("The keyframe absolute-position embedding is a video-stream parameter")

        existing = self._keyframes_embedding()
        if existing is not None and not existing.is_meta:
            return

        reference = self.patchify_proj.weight
        shape = existing.shape if existing is not None else (1, self.inner_dim)
        dtype = existing.dtype if existing is not None else reference.dtype
        self.use_keyframes_abs_pos_embedding = True
        self.keyframes_abs_pos_embedding = torch.nn.Parameter(torch.zeros(shape, dtype=dtype, device=reference.device))
