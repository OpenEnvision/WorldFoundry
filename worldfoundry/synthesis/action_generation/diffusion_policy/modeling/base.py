# Inference-only Diffusion Policy source retained in-tree.

import torch

from .module import ModuleAttrMixin


class BaseLowdimPolicy(ModuleAttrMixin):
    # ========= inference  ============
    # also as self.device and self.dtype for inference device transfer
    def predict_action(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        obs_dict:
            obs: B,To,Do
        return:
            action: B,Ta,Da
        To = 3
        Ta = 4
        T = 6
        |o|o|o|
        | | |a|a|a|a|
        |o|o|
        | |a|a|a|a|a|
        | | | | |a|a|
        """
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass
