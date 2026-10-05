"""
Refer to 
https://github.com/FoundationVision/LlamaGen
https://github.com/FoundationVision/VAR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from worldfoundry.core.model_loading.factory import instantiate_from_config
from src.Open_MAGVIT2.modules.util import SOSProvider


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class Net2NetTransformer(nn.Module):
    def __init__(self,
                transformer_config,
                first_stage_config,
                cond_stage_config,
                permuter_config=None,
                ckpt_path=None,
                ignore_keys=[],
                first_stage_key="image",
                cond_stage_key="depth",
                downsample_cond_size=-1,
                sos_token=0,
                unconditional=False,
                token_factorization=False,
                 ):
        super().__init__()

        self.be_unconditional = unconditional
        self.sos_token = sos_token
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.init_first_stage_from_ckpt(first_stage_config)
        self.init_cond_stage_from_ckpt(cond_stage_config)
        if permuter_config is None:
            permuter_config = {"target": "src.Open_MAGVIT2.modules.transformer.permuter.Identity"}
        self.permuter = instantiate_from_config(config=permuter_config)
        self.transformer = instantiate_from_config(config=transformer_config)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.downsample_cond_size = downsample_cond_size
        self.token_factorization = token_factorization

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def state_dict(self, *kwargs, destination=None, prefix='', keep_vars=False):
        return {k: v for k, v in super().state_dict(*kwargs, destination, prefix, keep_vars).items() if ("inception_model" not in k and "lpips_vgg" not in k and "lpips_alex" not in k)}

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

    def init_cond_stage_from_ckpt(self, config):
        if config == "__is_first_stage__":
            print("Using first stage also as cond stage.")
            self.cond_stage_model = self.first_stage_model
        elif config == "__is_unconditional__" or self.be_unconditional:
            print(f"Using no cond stage. Treating the model as unconditional with sos token {self.sos_token}.")
            self.be_unconditional = True
            self.cond_stage_key = self.first_stage_key
            self.cond_stage_model = SOSProvider(self.sos_token)
        else:
            model = instantiate_from_config(config)
            model = model.eval()
            model.train = disabled_train
            self.cond_stage_model = model

    def forward(self, x, c):
        # one step to produce the logits
        _, z_indices = self.encode_to_z(x)
        _, c_indices = self.encode_to_c(c)
        a_indices = z_indices

        if self.token_factorization: ## must be use
            a_indices_pre, a_indices_post = a_indices[0], a_indices[1]
            cz_indices = (a_indices_pre, a_indices_post, c_indices) ## AR inside

            target_pre = a_indices_pre
            target_post = a_indices_post
            
            logits, _ = self.transformer(cz_indices) #[B N 2 D]

            logits_pre, logits_post = logits[0], logits[1]

            logits = (logits_pre, logits_post)
            target = (target_pre, target_post)

        else:
            # target includes all sequence elements (no need to handle first one
            # differently because we are conditioning)
            target = z_indices
            # make the prediction
            cz_indices = (a_indices[:, :-1], c_indices) #not token factorization
            logits, _ = self.transformer(cz_indices)
            # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
            logits = logits[:, c_indices.shape[1]-1:]

        return logits, target

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, indices, _ = self.first_stage_model.encode(x)
        if isinstance(indices, tuple):
            indices_pre, indices_post = indices[0], indices[1]
            indices_pre = indices_pre.view(quant_z.shape[0], -1)
            indices_post = indices_post.view(quant_z.shape[0], -1)
            indices = (indices_pre, indices_post)
        else:
            indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def encode_to_c(self, c):
        if self.downsample_cond_size > -1:
            c = F.interpolate(c, size=(self.downsample_cond_size, self.downsample_cond_size))
        quant_c, _, [_,_,indices] = self.cond_stage_model.encode(c)
        if len(indices.shape) > 2:
            indices = indices.view(c.shape[0], -1)
        return quant_c, indices

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        if self.token_factorization: #using token factorization
            index_pre, index_post = index[0], index[1]
            # index_pre = self.permuter(index_pre, reverse=True)
            # index_post = self.permuter(index_post, reverse=True) #
            bhwc_pre = (zshape[0], zshape[2], zshape[3], self.transformer.config.factorized_bits[0])
            bhwc_post = (zshape[0], zshape[2], zshape[3], self.transformer.config.factorized_bits[1])
            # index_post -= 2**(zshape[1] // 2) ## no need since the seperate head
            # index_post[index_post < 0] = 0 #in case the overflow of index but usually not happended
            quant_pre = self.first_stage_model.quantize.get_codebook_entry(index_pre, bhwc_pre, order="pre")
            quant_post = self.first_stage_model.quantize.get_codebook_entry(index_post, bhwc_post, order="post")
            quant_z = torch.concat([quant_pre, quant_post], dim=1) # concate in the final dimension 
            x = self.first_stage_model.decode(quant_z)
        else:
            # index = self.permuter(index, reverse=True)
            bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
            quant_z = self.first_stage_model.quantize.get_codebook_entry(
                index, shape=bhwc)
            x = self.first_stage_model.decode(quant_z)
        return x

    def get_input(self, key, batch):
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        if len(x.shape) == 4:
            # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dtype == torch.double:
            x = x.float()
        return x

    def get_xc(self, batch, N=None):
        x = self.get_input(self.first_stage_key, batch)
        c = self.get_input(self.cond_stage_key, batch)
        if N is not None:
            x = x[:N]
            c = c[:N]
        return x, c
