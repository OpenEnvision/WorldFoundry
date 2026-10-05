from worldfoundry.base_models.diffusion_model.models.autoencoders.hunyuan_video import (
    load_hunyuan_video_causal3d,
)


class VAEWrapper:
    def __init__(self, vae):
        self.vae = vae

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self.vae, name)

    def encode(self, x):
        raise NotImplementedError

    def decode(self, latents):
        raise NotImplementedError


class MGVVAEWrapper(VAEWrapper):
    def __init__(self, vae):
        self.vae = vae
        self.vae.enable_tiling()
        self.vae.requires_grad_(False)
        self.vae.eval()

    def encode(self, x):
        x = self.vae.encode(x).latent_dist.sample()
        if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
            x = (x - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        else:
            x = x * self.vae.config.scaling_factor
        return x

    def decode(self, latents):
        if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
            latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        else:
            latents = latents / self.vae.config.scaling_factor
        return self.vae.decode(latents).sample


def get_mg_vae_wrapper(model_path, weight_dtype):
    path = model_path.removesuffix(".json")
    return MGVVAEWrapper(load_hunyuan_video_causal3d(path, dtype=weight_dtype))
