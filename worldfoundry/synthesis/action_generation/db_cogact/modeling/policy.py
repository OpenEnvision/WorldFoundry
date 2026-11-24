# Inference-only DB-CogACT source retained in-tree.
from __future__ import annotations

import numpy as np

from ..preprocessing import conversation as conversation_lib
from ..preprocessing.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from ..preprocessing.tokenization import tokenizer_image_token
from .policy_base import BasePolicy
from .types import ActionOutput, SamplingConfig


class CogACTPolicy(BasePolicy):
    """
    A-family policy for CogACT (and variants like HybridCogACT).

    observation keys (named camera format):
        "image/{cam}": PIL Image | ndarray | path str   — single sample
                       list of the above                — batch
        "prompt":      str (broadcast) | list[str]      — task instruction

    camera_order controls which image/* keys are used and in what order.
    """

    action_mode = "relative"
    state_used = False

    def select_action(
        self,
        observation: dict,
        sampling_config: SamplingConfig | None = None,
    ) -> list[ActionOutput]:
        batch_size = self._infer_batch_size(observation)
        obs = self._normalize_obs(observation, batch_size)
        return [self._infer_single({k: v[i] for k, v in obs.items()}, sampling_config) for i in range(batch_size)]

    def _infer_single(
        self,
        obs_single: dict,
        sampling_config: SamplingConfig | None,
    ) -> ActionOutput:
        text = obs_single["prompt"]

        img_keys = sorted(
            (k for k in obs_single if k.startswith("image/")),
            key=lambda k: int(k.split("/", 1)[1]),
        )
        pil_images = [self._load_images([obs_single[k]])[0] for k in img_keys]

        image_tensor = self._images_to_tensor(pil_images)

        conv = conversation_lib.conv_templates[self.model.config.chat_template].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + text)
        conv.append_message(conv.roles[1], " ")
        prompt = conv.get_prompt()

        execution_device = getattr(self.model, "_worldfoundry_execution_device", self.model.device)
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(execution_device)
        )

        sc = sampling_config or SamplingConfig()
        inference_args = {
            "cfg_scale": sc.cfg_scale,
            "num_ddim_steps": sc.num_steps,
            "action_norms": self.norm_stats,
            "seed": sc.seed,
        }

        actions = self.model.inference_action(input_ids, image_tensor, inference_args)
        return ActionOutput(actions=np.array(actions))
