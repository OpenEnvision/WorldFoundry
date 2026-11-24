import torch
from einops import rearrange

from src.distributions import DiagonalGaussianDistribution
from src.modules.ae_modules import Decoder, Encoder


class AutoencoderKL(torch.nn.Module):
    def __init__(
        self,
        ddconfig,
        lossconfig,
        embed_dim,
        use_quant_conv=True,
        ckpt_path=None,
        ignore_keys=[],  # noqa: B006 - Read-only default; retain the public signature.
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        test=False,
        logdir=None,
        input_dim=4,
        test_args=None,
    ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]

        if use_quant_conv:
            self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
            self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
            self.embed_dim = embed_dim

        self.use_quant_conv = use_quant_conv

        self.input_dim = input_dim
        self.test = test
        self.test_args = test_args
        self.logdir = logdir
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) is int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def init_from_ckpt(self, path, ignore_keys=[]):  # noqa: B006 - Read-only default; retain the public signature.
        sd = torch.load(path, map_location="cpu")
        try:
            self._cur_epoch = sd["epoch"]
            sd = sd["state_dict"]
        except BaseException:
            self._cur_epoch = "null"
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    # print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        # self.load_state_dict(sd, strict=True)
        print(f"Restored from {path}")

    def encode(self, x, **kwargs):
        h = self.encoder(x)
        moments = h
        if self.use_quant_conv:
            moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z, **kwargs):
        if self.use_quant_conv:
            z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if x.dim() == 5 and self.input_dim == 4:
            b, c, t, h, w = x.shape
            self.b = b
            self.t = t
            x = rearrange(x, "b c t h w -> (b t) c h w")

        return x


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        # TODO: Should be true by default but check to not break older stuff
        self.vq_interface = vq_interface
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x
