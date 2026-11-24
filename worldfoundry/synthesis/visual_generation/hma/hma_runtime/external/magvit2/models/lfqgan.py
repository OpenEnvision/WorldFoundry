"""
Modification of Open-MAGVIT2 code, including adding gradient accumulation during training, using VQConfig,
removing hardcoded arguments and removing unnecessary code. Only the inference tokenizer (encoder, LFQ
quantizer and decoder) is kept.
"""

import torch
from torch import nn

from ..config import VQConfig
from ..modules.diffusionmodules.improved_model import Decoder, Encoder
from ..modules.vqvae.lookup_free_quantize import LFQ


class VQModel(nn.Module):
    def __init__(self, config: VQConfig, ckpt_path=None):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.quantize = LFQ(config)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        # Released tokenizer checkpoints also hold the training loss, discriminator and EMA state.
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x, **kwargs):
        h = self.encoder(x)
        (quant, emb_loss, info), loss_breakdown = self.quantize(h, return_loss_breakdown=True, **kwargs)
        ### using token factorization the info is a tuple (each for embedding)
        return quant, emb_loss, info, loss_breakdown

    def decode(self, quant):
        dec = self.decoder(quant)
        return dec
