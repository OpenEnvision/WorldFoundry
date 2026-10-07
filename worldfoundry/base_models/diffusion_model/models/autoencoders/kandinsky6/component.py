"""Video/audio codec component for Kandinsky-6 joint generation."""


class Kandinsky6Decoder:
    def __init__(self, video, audio, vocoder):
        self.video = video
        self.audio = audio
        self.vocoder = vocoder

    def decode(self, latents, request):
        return self.video.decode(latents).sample


def build_kandinsky6_decoder(context):
    from ....loaders.kandinsky6 import component_config
    from .vae_video import build_vae
    from .vae_audio import build_audio_vae, build_vocoder

    cfg, device = component_config(context)
    video = build_vae(cfg.paths.vae, device=device)
    audio = build_audio_vae(**cfg.audio_vae.model_dump(), device=device)
    vocoder = build_vocoder(**cfg.vocoder.model_dump(), device=device)
    return Kandinsky6Decoder(video, audio, vocoder)
