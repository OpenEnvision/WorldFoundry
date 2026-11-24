"""Low-dimensional conditioning masks used by Diffusion Policy."""

from __future__ import annotations

import torch

from .module import ModuleAttrMixin


class LowdimMaskGenerator(ModuleAttrMixin):
    """Reproduce the conditioning mask module stored in official checkpoints."""

    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        *,
        max_n_obs_steps: int = 2,
        fix_obs_steps: bool = True,
        action_visible: bool = False,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.max_n_obs_steps = max_n_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.action_visible = action_visible

    @torch.no_grad()
    def forward(self, shape: tuple[int, int, int], seed: int | None = None) -> torch.Tensor:
        batch_size, horizon, feature_dim = shape
        if feature_dim != self.action_dim + self.obs_dim:
            raise ValueError("LowdimMaskGenerator feature dimension must equal action_dim + obs_dim")

        device = self.device
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)

        dimension_mask = torch.zeros(shape, dtype=torch.bool, device=device)
        action_dimensions = dimension_mask.clone()
        action_dimensions[..., : self.action_dim] = True
        observation_dimensions = ~action_dimensions

        if self.fix_obs_steps:
            observation_steps = torch.full((batch_size,), self.max_n_obs_steps, device=device)
        else:
            observation_steps = torch.randint(
                low=1,
                high=self.max_n_obs_steps + 1,
                size=(batch_size,),
                generator=generator,
                device=device,
            )

        steps = torch.arange(horizon, device=device).reshape(1, horizon).expand(batch_size, horizon)
        observation_mask = (observation_steps > steps.T).T.reshape(batch_size, horizon, 1).expand(shape)
        observation_mask = observation_mask & observation_dimensions

        if not self.action_visible:
            return observation_mask

        action_steps = torch.maximum(
            observation_steps - 1,
            torch.tensor(0, dtype=observation_steps.dtype, device=device),
        )
        action_mask = (action_steps > steps.T).T.reshape(batch_size, horizon, 1).expand(shape)
        action_mask = action_mask & action_dimensions
        return observation_mask | action_mask
