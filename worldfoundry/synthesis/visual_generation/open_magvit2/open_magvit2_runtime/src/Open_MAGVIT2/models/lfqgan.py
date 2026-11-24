from collections import OrderedDict

import torch
import torch.nn as nn

from worldfoundry.synthesis.visual_generation.open_magvit2.open_magvit2_runtime.src.Open_MAGVIT2.modules.diffusionmodules.improved_model import (
    Decoder,
    Encoder,
)
from worldfoundry.synthesis.visual_generation.open_magvit2.open_magvit2_runtime.src.Open_MAGVIT2.modules.vqvae.lookup_free_quantize import (
    LFQ,
)


class VQModel(nn.Module):
    def __init__(
        self,
        ddconfig,
        ## Quantize Related
        n_embed,
        embed_dim,
        sample_minimization_weight,
        batch_maximization_weight,
        lossconfig=None,
        ckpt_path=None,
        ignore_keys=[],  # noqa: B006 - Keeps the upstream default object and signature.
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        token_factorization=False,
        stage=None,
        factorized_bits=[9, 9],  # noqa: B006 - Keeps the upstream default object and signature.
        model_type="en18",
        **_unused,
    ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig, model_type=model_type)
        self.decoder = Decoder(**ddconfig)
        self.quantize = LFQ(
            dim=embed_dim,
            codebook_size=n_embed,
            sample_minimization_weight=sample_minimization_weight,
            batch_maximization_weight=batch_maximization_weight,
            token_factorization=token_factorization,
            factorized_bits=factorized_bits,
        )

        if colorize_nlabels is not None:
            assert type(colorize_nlabels) is int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, stage=stage)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def load_state_dict(self, *args, strict=False):
        """
        Resume not strict loading
        """
        return super().load_state_dict(*args, strict=strict)

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        """
        filter out the non-used keys
        """
        return {
            k: v
            for k, v in super().state_dict(*args, destination, prefix, keep_vars).items()
            if ("inception_model" not in k and "lpips_vgg" not in k and "lpips_alex" not in k)
        }

    def init_from_ckpt(self, path, ignore_keys=[], stage="transformer"):  # noqa: B006 - Keeps the upstream default object and signature.
        sd = torch.load(path, map_location="cpu")["state_dict"]
        new_params = OrderedDict()
        if stage == "transformer":
            for k, v in sd.items():
                if "encoder" in k or "decoder" in k:
                    new_params[k.replace("model_ema.", "")] = v
        self.load_state_dict(new_params or sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        (quant, emb_loss, info), loss_breakdown = self.quantize(h, return_loss_breakdown=True)
        return quant, emb_loss, info, loss_breakdown

    def decode(self, quant):
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _, loss_break = self.encode(input)
        dec = self.decode(quant)
        return dec, diff, loss_break

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).contiguous()
        return x.float()
