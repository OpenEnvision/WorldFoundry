# Inference-only Diffusion Policy source retained in-tree.

import numpy as np
import torch

from worldfoundry.base_models.llm_mllm_core.action.diffusion_policy import (
    DictOfTensorMixin,
)


class LinearNormalizer(DictOfTensorMixin):
    def __call__(self, x: dict | torch.Tensor | np.ndarray) -> torch.Tensor:
        return self.normalize(x)

    def __getitem__(self, key: str):
        return SingleFieldLinearNormalizer(self.params_dict[key])

    def _normalize_impl(self, x, forward=True):
        if isinstance(x, dict):
            result = {}
            for key, value in x.items():
                params = self.params_dict[key]
                result[key] = _normalize(value, params, forward=forward)
            return result
        else:
            if "_default" not in self.params_dict:
                raise RuntimeError("Not initialized")
            params = self.params_dict["_default"]
            return _normalize(x, params, forward=forward)

    def normalize(self, x: dict | torch.Tensor | np.ndarray) -> torch.Tensor:
        return self._normalize_impl(x, forward=True)

    def unnormalize(self, x: dict | torch.Tensor | np.ndarray) -> torch.Tensor:
        return self._normalize_impl(x, forward=False)


class SingleFieldLinearNormalizer(DictOfTensorMixin):
    def normalize(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=True)

    def unnormalize(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=False)

    def __call__(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        return self.normalize(x)


def _normalize(x, params, forward=True):
    assert "scale" in params
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params["scale"]
    offset = params["offset"]
    x = x.to(device=scale.device, dtype=scale.dtype)
    src_shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    x = x * scale + offset if forward else (x - offset) / scale
    x = x.reshape(src_shape)
    return x
