from typing import Dict, List, Optional, Tuple, Union

import torch
from einops import rearrange
from omegaconf import ListConfig, OmegaConf
from pytorch_lightning import LightningModule
from vwm.modules import UNCONDITIONAL_CONFIG
from vwm.modules.autoencoding.temporal_ae import VideoDecoder
from vwm.modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from vwm.util import default, disabled_train, get_obj_from_str, instantiate_from_config


class DiffusionEngine(LightningModule):
    def __init__(
            self,
            network_config,
            denoiser_config,
            first_stage_config,
            conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
            sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
            network_wrapper: Union[None, str] = None,
            scale_factor: float = 1.0,
            disable_first_stage_autocast=False,
            input_key: str = "img",
            compile_model: bool = False,
            en_and_decode_n_samples_a_time: Optional[int] = None,
            n_context_frames: int = 5
    ):
        super(DiffusionEngine, self).__init__()
        self.input_key = input_key
        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            model, compile_model=compile_model
        )

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = instantiate_from_config(sampler_config) if sampler_config is not None else None
        self.conditioner = instantiate_from_config(default(conditioner_config, UNCONDITIONAL_CONFIG))
        self._init_first_stage(first_stage_config)

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast

        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
        self.n_context_frames = n_context_frames
        self.num_frames = n_context_frames + 1

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def get_input(self, batch):
        # Image tensors should be scaled to -1 ... 1 and in bchw format
        input_shape = batch[self.input_key].shape
        if len(input_shape) != 4:  # Is an image sequence
            assert input_shape[1] == self.num_frames
            batch[self.input_key] = rearrange(batch[self.input_key], "b t c h w -> (b t) c h w")
        return batch[self.input_key]

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = z / self.scale_factor
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for current_z in z.split(n_samples, dim=0):
                if isinstance(self.first_stage_model.decoder, VideoDecoder):
                    kwargs = {"timesteps": current_z.shape[0]}
                else:
                    kwargs = {}
                out = self.first_stage_model.decode(current_z, **kwargs)
                all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for current_x in x.split(n_samples, dim=0):
                out = self.first_stage_model.encode(current_x)
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = z * self.scale_factor
        return z

    @torch.no_grad()
    def sample(
            self,
            cond: Dict,
            x_ori: torch.Tensor,
            uc: Union[Dict, None] = None,
            N: int = 10,
            shape: Union[None, Tuple, List] = None,
            **kwargs
    ):
        randn = torch.randn(N, *shape).to(self.device)

        denoiser = lambda input, sigma, c: self.denoiser(self.model, input, sigma, c, **kwargs)

        samples = self.sampler(denoiser, randn, cond, x_ori=x_ori, uc=uc)
        return samples
