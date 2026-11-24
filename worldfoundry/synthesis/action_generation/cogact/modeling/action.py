# Inference-only CogACT source retained in-tree.
"""
action_model.py

"""

from torch import nn

from worldfoundry.base_models.llm_mllm_core.action import cogact_diffusion as gd
from worldfoundry.base_models.llm_mllm_core.action.cogact_diffusion import (
    create_diffusion,
)

from .dit import DiT


# Create model sizes of ActionModels
def DiT_S(**kwargs):
    return DiT(depth=6, hidden_size=384, num_heads=4, **kwargs)


def DiT_B(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)


def DiT_L(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)


# Model size
DiT_models = {"DiT-S": DiT_S, "DiT-B": DiT_B, "DiT-L": DiT_L}


# Create ActionModel
class ActionModel(nn.Module):
    def __init__(
        self,
        token_size,
        model_type,
        in_channels,
        future_action_window_size,
        past_action_window_size,
        diffusion_steps=100,
        noise_schedule="squaredcos_cap_v2",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.noise_schedule = noise_schedule
        # GaussianDiffusion implements the reverse DDPM/DDIM sampling process.
        self.diffusion_steps = diffusion_steps
        self.diffusion = create_diffusion(
            timestep_respacing="",
            noise_schedule=noise_schedule,
            diffusion_steps=self.diffusion_steps,
            sigma_small=True,
            learn_sigma=False,
        )
        self.ddim_diffusion = None
        if self.diffusion.model_var_type in [gd.ModelVarType.LEARNED, gd.ModelVarType.LEARNED_RANGE]:
            learn_sigma = True
        else:
            learn_sigma = False
        self.past_action_window_size = past_action_window_size
        self.future_action_window_size = future_action_window_size
        self.net = DiT_models[model_type](
            token_size=token_size,
            in_channels=in_channels,
            class_dropout_prob=0.1,
            learn_sigma=learn_sigma,
            future_action_window_size=future_action_window_size,
            past_action_window_size=past_action_window_size,
        )

    # Create DDIM sampler
    def create_ddim(self, ddim_step=10):
        self.ddim_diffusion = create_diffusion(
            timestep_respacing="ddim" + str(ddim_step),
            noise_schedule=self.noise_schedule,
            diffusion_steps=self.diffusion_steps,
            sigma_small=True,
            learn_sigma=False,
        )
        return self.ddim_diffusion
