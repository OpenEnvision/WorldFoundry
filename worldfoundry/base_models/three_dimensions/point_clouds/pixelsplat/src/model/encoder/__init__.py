from .encoder import Encoder as Encoder
from .encoder_epipolar import EncoderEpipolar, EncoderEpipolarCfg

ENCODERS = {
    "epipolar": (EncoderEpipolar, None),
}

EncoderCfg = EncoderEpipolarCfg


def get_encoder(cfg: EncoderCfg):
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    return encoder, None
